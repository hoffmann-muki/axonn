# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Distributed training example: VGG16 on Caltech-256 using AxoNN.

This script demonstrates a minimal distributed training loop that uses
AxoNN to set up data-parallel process groups and to shard the input
dataset across ranks.

Notes:
- Use ``--dataset-root`` to point to Caltech-256 or ``--download`` to
    fetch the dataset on rank 0.
- This example logs aggregated gradient-message sizes when enabled via
    ``--log-grad-sizes``; the logging uses AxoNN's data-parallel group
    and will be skipped if AxoNN is not initialized.
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
def apply_topk_sparsification(model, topk_ratio: float = 0.05) -> None:
    """Sparsify gradients in-place by keeping the largest magnitudes.

    For each parameter, keep the largest ``topk_ratio`` fraction of
    gradient elements (by absolute value) and zero the remainder.
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


def log_grad_message_sizes(model, top_n: int = 10) -> None:
    """Compute and report per-data-parallel-rank gradient-message sizes.

    This computes the approximate number of bytes that would be communicated
    during an all-reduce of gradients for ``model``. The function requires
    AxoNN's data-parallel group (``ax.comm_handle.data_parallel_group``)
    and will skip logging if AxoNN is not initialized.
    """
    # Collect local per-parameter gradient sizes (bytes)
    per_rank_info = []
    local_total = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        nbytes = int(p.grad.numel() * p.grad.element_size())
        local_total += nbytes
        per_rank_info.append((name, tuple(p.grad.shape), nbytes))

    # Require AxoNN's data-parallel group; do not fall back to global world group
    if not (dist.is_initialized() and hasattr(ax, "comm_handle") and getattr(ax.comm_handle, "data_parallel_group", None) is not None):
        if dist.is_initialized() and dist.get_rank() == 0:
            print("AxoNN data-parallel group unavailable; skipping grad-size logging")
        return

    group = ax.comm_handle.data_parallel_group
    group_size = ax.comm_handle.G_data
    rank_in_group = ax.comm_handle.data_parallel_rank

    # Build a 1-element tensor on the same device as model parameters for collectives
    try:
        device = next(model.parameters()).device
        tensor_device = device if device.type == "cuda" else torch.device("cpu")
    except StopIteration:
        tensor_device = torch.device("cpu")

    local = torch.tensor([local_total], dtype=torch.long, device=tensor_device)
    gathered = [torch.zeros_like(local) for _ in range(group_size)]
    dist.all_gather(gathered, local, group=group)
    gathered_ints = [int(x.item()) for x in gathered]

    # Print per-group-rank totals from the group's rank 0
    if rank_in_group == 0:
        print("[group-rank 0] per-group-rank allreduce bytes (approx):")
        for i, val in enumerate(gathered_ints):
            print(f"  rank {i}: {val} bytes ({val/1024**2:.3f} MB)")


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
                            num_workers=None,
                            optimizer_name="adamw",
                            log_grad_sizes: bool = False):
    """
    Distributed training routine for VGG16 on Caltech-256 using AxoNN.

    Expects `dataset_root` to point to the Caltech-256 dataset root as
    required by `torchvision.datasets.Caltech256`.
    """
    # Configuration
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = num_gpus * batch_size_per_gpu

    if base_lr is None:
        # default LR depends on optimizer choice: SGD uses 0.1 scale, AdamW uses 1e-3 scale
        if optimizer_name.lower() == 'sgd':
            base_lr = 0.1 * (global_batch_size / 256)
        else:
            base_lr = 1e-3 * (global_batch_size / 256)

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

    # Try to infer number of classes from the dataset
    try:
        tmp_ds = load_caltech256_dataset(dataset_root, split=split, transform=None)
        inferred = None
        if hasattr(tmp_ds, 'classes'):
            inferred = len(tmp_ds.classes)
        elif hasattr(tmp_ds, 'categories'):
            inferred = len(tmp_ds.categories)
        if inferred is not None and inferred > 0 and inferred != num_classes:
            if rank == 0:
                print(f"Adjusting num_classes from {num_classes} to {inferred} based on dataset")
            num_classes = inferred
    except Exception:
        # dataset may not be available yet or loader may raise; continue with provided num_classes
        pass

    # Instantiate VGG16 using modern weights API
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights).cuda()
    # Adapt classifier for target number of classes
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes).cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-4)

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
        train_correct = 0
        train_total = 0

        for batch_idx, (x, y) in enumerate(tqdm(train_dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}/{num_epochs}")):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            # optionally log approximate gradient all-reduce message sizes (aggregated, rank 0)
            if log_grad_sizes:
                try:
                    log_grad_message_sizes(model, top_n=8)
                except Exception:
                    # don't fail training if logging has issues
                    pass

            # compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if topk_ratio is not None and 0.0 < topk_ratio < 1.0:
                apply_topk_sparsification(model, topk_ratio=topk_ratio)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            batch_loss = float(loss.item())
            epoch_loss += batch_loss

            # training accuracy accumulation
            preds = logits.argmax(dim=1)
            train_correct += int((preds == y).sum().item())
            train_total += x.size(0)

            # print per-batch info including grad norm (rank 0 only)
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx+1}: loss = {batch_loss:.6f} | LR = {current_lr:.6e} | grad_norm = {total_norm:.6f}")

        if rank == 0:
            avg_loss = epoch_loss / max(1, len(train_dataloader))
            train_acc = 100.0 * train_correct / max(1, train_total)
            print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}, Train Acc = {train_acc:.2f}%, Duration = {time.time()-epoch_start:.2f}s")

        # Validation (optional): try to load 'val' split and evaluate
        try:
            val_transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_ds = load_caltech256_dataset(dataset_root, split="val", transform=val_transform)
            val_loader = DataLoader(val_ds, batch_size=global_batch_size // max(1, num_gpus), shuffle=False, num_workers=num_workers or 0)

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.cuda(), vy.cuda()
                    v_logits = model(vx)
                    v_preds = v_logits.argmax(dim=1)
                    val_correct += int((v_preds == vy).sum().item())
                    val_total += vx.size(0)

            val_acc = 100.0 * val_correct / max(1, val_total)
            if rank == 0:
                print(f"Validation Acc = {val_acc:.2f}% ({val_correct}/{val_total})")
            model.train()
        except Exception:
            # skip validation if loader doesn't support 'val' or data not present
            if rank == 0:
                print("Validation skipped (no 'val' split or loader error)")

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
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd","adamw"], help="Optimizer to use (sgd or adamw)")
    parser.add_argument("--download", action="store_true", help="Download Caltech-256 into --dataset-root (rank 0 only)")
    parser.add_argument("--log-grad-sizes", action="store_true", help="Enable aggregated gradient-message-size logging (global total printed on rank 0)")
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
            optimizer_name=args.optimizer,
            log_grad_sizes=args.log_grad_sizes,
        )


if __name__ == "__main__":
    main()
