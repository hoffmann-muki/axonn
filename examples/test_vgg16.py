# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Distributed training example: VGG16 on Caltech-256 using AxoNN.
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
from torch.utils.data import DataLoader

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


def load_caltech256_dataset(root, split="train", transform=None):
    """Return the Caltech-256 dataset using torchvision's loader."""
    return datasets.Caltech256(root=root, transform=transform, download=False)


def ensure_caltech256_download(root):
    """Download Caltech-256 on rank 0 and wait on other ranks."""
    marker = os.path.join(root, ".caltech256_download_complete")
    if os.path.exists(marker):
        return

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        os.makedirs(root, exist_ok=True)
        print(f"Rank 0 downloading Caltech-256 into {root} (this may take a while)")
        datasets.Caltech256(root=root, download=True)
        # create sentinel
        try:
            with open(marker, "w") as f:
                f.write("ok")
        except Exception:
            pass
    else:
        # wait for marker to appear
        while not os.path.exists(marker):
            time.sleep(1)

def train_vgg16_distributed(topk_ratio=0.0,
                            batch_size_per_gpu=32,
                            num_epochs=30,
                            base_lr=None,
                            pretrained=False,
                            dataset_root=None,
                            split="train",
                            num_classes=256,
                            num_workers=None):
    """
    Distributed training routine for VGG16 on Caltech-256 using AxoNN.

    Expects `dataset_root` to point to the Caltech-256 dataset root as
    required by `torchvision.datasets.Caltech256`.
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

    # VGG input transforms (224x224, ImageNet normalization)
    train_transform = transforms.Compose([
        # Ensure images are RGB to avoid grayscale corruption from some datasets
        transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dist.barrier()

    # Load dataset on all ranks
    train_dataset = load_caltech256_dataset(dataset_root, split=split, transform=train_transform)

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
    parser = argparse.ArgumentParser(description="Train VGG16 on Caltech-256 with AxoNN")
    parser.add_argument("--topk", type=float, default=0.0, help="Fraction of gradients to keep via top-k sparsification (0 disables)")
    parser.add_argument("--batch-size-per-gpu", type=int, default=32, help="Per-GPU micro-batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate (linear scaling if omitted)")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained VGG16 weights")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to Caltech-256 root (as expected by torchvision.datasets.Caltech256)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name (train or val)")
    parser.add_argument("--num-classes", type=int, default=256, help="Number of target classes")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers per process")
    parser.add_argument("--download", action="store_true", help="Download Caltech-256 into --dataset-root (rank 0 only)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a CPU-only dry-run: validate dataset and model instantiation without distributed init or GPUs")

    args = parser.parse_args()
    if args.download:
        ensure_caltech256_download(args.dataset_root)
        print("Caltech-256 download complete (--download); exiting.")
        return
    if args.dry_run:
        # Dry-run: validate dataset and model on CPU
        print("Dry-run: validating dataset and model on CPU")
        tr = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ds = load_caltech256_dataset(args.dataset_root, split=args.split, transform=tr)
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
            split=args.split,
            num_classes=args.num_classes,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
