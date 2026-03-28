# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Distributed training example using AxoNN's data parallelism with VGG16 on Open Images.

This example demonstrates how to:
    - Initialize AxoNN with data parallelism across multiple GPUs
    - Construct a distributed dataloader that shards data across ranks
    - Implement a distributed training loop with gradient synchronization via NCCL
    - Load Open Images (prefer torchvision.OpenImages; falls back to ImageFolder)
"""

import os
import time

import argparse

import torch
import torch.distributed as dist
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.models import VGG16_Weights
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from axonn import axonn as ax

@torch.no_grad()
def apply_topk_sparsification(model, topk_ratio=0.05):
    """
    Sparsifies gradients by keeping only the top-k magnitude values and
    zeroing out the rest. `topk_ratio` is the fraction of elements to keep
    (e.g. 0.05 keeps the top 5% of gradient magnitudes).
    """
    for p in model.parameters():
        if p.grad is None:
            continue

        grad_flat = p.grad.view(-1)
        total_elements = grad_flat.numel()

        k = max(1, int(total_elements * topk_ratio))

        _, indices = torch.topk(grad_flat.abs(), k)

        topk_values = grad_flat[indices]
        new_grad = torch.zeros_like(grad_flat)
        new_grad.scatter_(0, indices, topk_values)

        p.grad.copy_(new_grad.view(p.grad.shape))


def load_open_images_dataset(root, split="train", transform=None):
    """
    Load Open Images dataset.

    This prefers `torchvision.datasets.OpenImages` when available. If the
    user has exported Open Images into a class-organized ImageFolder layout
    (root/<split>/<class>/*.jpg) this function will load via `ImageFolder`.

    Arguments:
        root (str): dataset root directory or parent folder containing split subdirs
        split (str): 'train' or 'validation' (or dataset-specific)
        transform: torchvision transforms to apply

    Returns:
        A torch Dataset instance.
    """
    # prefer torchvision.OpenImages if available and user points to original layout
    OpenImages = getattr(datasets, "OpenImages", None)

    if OpenImages is not None and os.path.isdir(root) and any(name.lower().startswith("openimages") for name in os.listdir(root)):
        # best-effort: user has the raw OpenImages checkout under `root`
        return OpenImages(root=root, split=split, transform=transform)

    # fallback: expect ImageFolder layout at root/<split>/class_name/*.jpg
    split_path = os.path.join(root, split)
    if os.path.isdir(split_path):
        return datasets.ImageFolder(split_path, transform=transform)

    raise RuntimeError(
        f"OpenImages not found at {root} and no ImageFolder at {split_path}. "
        "Prepare data as OpenImages or organize images under <root>/<split>/<class>/..."
    )

def train_vgg16_distributed(topk_ratio=0.0,
                            batch_size_per_gpu=32,
                            num_epochs=30,
                            base_lr=None,
                            pretrained=False,
                            dataset_root="./open_images",
                            num_classes=600,
                            num_workers=None):
    """
    Distributed training routine for VGG16 on Open Images using AxoNN.

    Expects `dataset_root` to contain the dataset, either in OpenImages
    layout (if torchvision.OpenImages is available) or as
    `dataset_root/<split>/<class_name>/*.jpg` for ImageFolder.
    """
    # Configuration
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = num_gpus * batch_size_per_gpu

    if base_lr is None:
        base_lr = 0.1 * (global_batch_size / 256)

    if num_workers is None:
        num_workers = min(8, (os.cpu_count() or 4))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    ax.init(G_data=num_gpus, G_inter=1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Initialized distributed training on {world_size} GPUs")
        print(f"Global batch size: {global_batch_size}")
        print(f"Base LR: {base_lr:.6f}")

    # Instantiate VGG16 using modern weights API
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights).cuda()
    # Adapt classifier for target number of classes
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes).cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)

    # Build transforms for VGG (224x224 ImageNet-style)
    train_transform = transforms.Compose([
        # Ensure images are RGB to avoid grayscale corruption from some datasets
        transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Only rank 0 may perform dataset setup/download
    if rank == 0:
        print("Preparing Open Images dataset (rank 0)...")
        # attempt a dry-run load to trigger any checks/downloads if needed
        try:
            _ = load_open_images_dataset(dataset_root, split="train", transform=train_transform)
        except Exception as e:
            print(f"Dataset preparation error on rank 0: {e}")
            raise

    # synchronize so all ranks wait for rank 0
    dist.barrier()

    # load dataset on all ranks
    train_dataset = load_open_images_dataset(dataset_root, split="train", transform=train_transform)

    train_dataloader = ax.create_dataloader(
        dataset=train_dataset,
        global_batch_size=global_batch_size,
        micro_batch_size=batch_size_per_gpu,
        num_workers=num_workers,
    )

    # Scheduler
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, steps_per_epoch // 10)

    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(tqdm(train_dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}/{num_epochs}")):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            if topk_ratio is not None and 0.0 < topk_ratio < 1.0:
                apply_topk_sparsification(model, topk_ratio=topk_ratio)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            batch_loss = float(loss.item())
            epoch_loss += batch_loss

            if rank == 0 and batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx+1}: loss = {batch_loss:.6f} | LR = {current_lr:.6e}")

        if rank == 0:
            avg_loss = epoch_loss / max(1, len(train_dataloader))
            print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}, Duration = {time.time()-epoch_start:.2f}s")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train VGG16 on Open Images with AxoNN")
    parser.add_argument("--topk", type=float, default=0.0, help="Fraction of gradients to keep via top-k sparsification (0 disables)")
    parser.add_argument("--batch-size-per-gpu", type=int, default=32, help="Per-GPU micro-batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate (linear scaling if omitted)")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained VGG16 weights")
    parser.add_argument("--dataset-root", type=str, default="./open_images", help="Path to OpenImages root or ImageFolder layout")
    parser.add_argument("--num-classes", type=int, default=600, help="Number of target classes")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers per process")
    parser.add_argument("--dry-run", action="store_true", help="Perform a CPU-only dry-run: validate dataset and model instantiation without distributed init or GPUs")

    args = parser.parse_args()
    if args.dry_run:
        # Validate dataset loading and model instantiation on CPU without distributed init
        print("Running dry-run: validating dataset and model on CPU")
        tr = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ds = load_open_images_dataset(args.dataset_root, split="train", transform=tr)
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=(args.num_workers or 0))
        # instantiate model on CPU
        weights = VGG16_Weights.IMAGENET1K_V1 if args.pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, args.num_classes)

        # Iterate a couple of batches
        for i, (x, y) in enumerate(loader):
            print(f"Batch {i}: x.shape={getattr(x, 'shape', None)}, y.shape={getattr(y, 'shape', None)}")
            if i >= 2:
                break
        print("Dry-run completed successfully")
    else:
        train_vgg16_distributed(
            topk_ratio=args.topk,
            batch_size_per_gpu=args.batch_size_per_gpu,
            num_epochs=args.epochs,
            base_lr=args.lr,
            pretrained=args.pretrained,
            dataset_root=args.dataset_root,
            num_classes=args.num_classes,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
