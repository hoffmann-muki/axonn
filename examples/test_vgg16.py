# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Distributed training example using AxoNN's data parallelism with VGG16.

This example demonstrates how to:
  - Initialize AxoNN with data parallelism across multiple GPUs
  - Construct a distributed dataloader that shards data across ranks
  - Implement a distributed training loop with gradient synchronization via NCCL
"""

import os
import time

import torch
import torch.distributed as dist
import argparse
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset

from axonn import axonn as ax

from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

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


def load_tiny_imagenet_dataset(split="train", local_cache_dir=None):
    """
    Load Tiny ImageNet dataset using Hugging Face `datasets`.

    This function will prefer a local cache directory when provided or when
    a conventional local cache path exists inside the repository
    (`./zh-plus___tiny-imagenet/default/0.0.0`). If no local cache is found
    it falls back to loading from the Hugging Face Hub.

    Arguments:
        split (str): 'train' for training set, 'valid' for validation set
        local_cache_dir (str|None): Optional path to a local HF datasets cache

    Returns:
        Dataset with images and labels, configured with standard ImageNet transforms
    """
    # Prefer an explicitly supplied cache dir, else check common local path
    if local_cache_dir is None:
        local_cache_dir = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_DATASETS_CACHE")
        if not local_cache_dir:
            candidate = os.path.join(os.getcwd(), "zh-plus___tiny-imagenet", "default", "0.0.0")
            if os.path.exists(candidate):
                local_cache_dir = candidate

    # Load dataset using local cache when available to avoid downloading
    if local_cache_dir and os.path.exists(local_cache_dir):
        dataset = load_dataset("zh-plus/tiny-imagenet", split=split, cache_dir=local_cache_dir)
    else:
        dataset = load_dataset("zh-plus/tiny-imagenet", split=split)
    
    # Define image transformations
    # For training: use RandomResizedCrop + horizontal flip + normalization.
    # For validation: resize and center-crop to 224 then normalize.
    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # Create a wrapper to apply transforms on-the-fly
    class TinyImageNetWrapper(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item["image"]
            label = item["label"]
            
            # Convert to RGB to handle grayscale images
            # (Tiny ImageNet contains both RGB and grayscale images)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    return TinyImageNetWrapper(dataset, transform=transform)

def train_vgg16_distributed(topk_ratio=0.0,
                            batch_size_per_gpu=32,
                            num_epochs=30,
                            base_lr=None,
                            pretrained=False,
                            num_workers=None):
    """
    Distributed training routine for VGG16 on Tiny ImageNet.
    
    Requires:
      - WORLD_SIZE environment variable set to the number of processes
      - Launch via torchrun or mpirun with proper rank environment variables
    """
    # Configuration (defaults chosen for stable training)
    num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = num_gpus * batch_size_per_gpu

    # Learning rate: if not supplied, use linear-scaling heuristic for SGD
    if base_lr is None:
        base_lr = 0.1 * (global_batch_size / 256)

    # Reasonable default for data loader workers
    if num_workers is None:
        num_workers = min(8, (os.cpu_count() or 4))

    # Initialize torch.distributed process group (required before AxoNN initialization)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Initialize AxoNN with data parallelism configuration
    # G_data specifies the number of data-parallel ranks
    # G_inter=1 indicates no pipeline parallelism
    ax.init(G_data=num_gpus, G_inter=1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Rank 0 prints training configuration
    if rank == 0:
        print(f"Initialized distributed training on {world_size} GPUs")
        print(f"Global batch size: {global_batch_size}")

    # Instantiate model and training components
    model = models.vgg16(pretrained=pretrained).cuda()

    # Use standard cross-entropy for initial experiments
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    # SGD with momentum and weight decay is a robust baseline for image
    # classification when training from scratch; scale lr with batch size.
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4
    )

    # NOTE: scheduler will be created after the dataloader is constructed
    # so that we can compute warmup/total steps from the number of batches.
    scheduler = None

    # Construct distributed dataloader
    # ax.create_dataloader handles data sharding across data-parallel ranks
    if rank == 0:
        print("Loading Tiny ImageNet training dataset...")

    # Prefer a local HF datasets cache when present. The code checks the
    # `HF_DATASETS_CACHE` environment variable and then a repo-local candidate
    # path (`./zh-plus___tiny-imagenet/default/0.0.0`). If neither exists the
    # loader will fall back to the HF Hub.
    local_cache = os.environ.get("HF_DATASETS_CACHE") or os.environ.get("HF_DATASETS_CACHE")
    if not local_cache:
        candidate = os.path.join(os.getcwd(), "zh-plus___tiny-imagenet", "default", "0.0.0")
        if os.path.exists(candidate):
            local_cache = candidate

    if rank == 0:
        if local_cache and os.path.exists(local_cache):
            print(f"Using local Tiny ImageNet cache at {local_cache}")
        else:
            print("No local Tiny ImageNet cache detected; will download from HF Hub if needed")

    train_dataset = load_tiny_imagenet_dataset(split="train", local_cache_dir=local_cache)
    
    train_dataloader = ax.create_dataloader(
        dataset=train_dataset,
        global_batch_size=global_batch_size,
        micro_batch_size=batch_size_per_gpu,
        num_workers=num_workers,
    )

    # Build scheduler based on steps per epoch (batches per epoch)
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_epochs
    # Warmup for a small fraction of the first epoch (10% of an epoch)
    warmup_steps = max(1, steps_per_epoch // 10)

    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(tqdm(
            train_dataloader,
            disable=(rank != 0),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )):
            # Move data to GPU
            x, y = x.cuda(), y.cuda()

            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            # Backward pass (gradient synchronization occurs automatically via NCCL)
            loss.backward()

            # Apply gradient sparsification (keep fraction `topk_ratio`)
            if topk_ratio is not None and topk_ratio > 0.0 and topk_ratio < 1.0:
                apply_topk_sparsification(model, topk_ratio=topk_ratio)

            # Gradient clipping to help with stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Parameter update
            optimizer.step()

            # Update the learning rate every batch
            scheduler.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Log per-batch loss for rank 0
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx + 1}: loss = {batch_loss:.6f} | LR = {current_lr:.6e}")

        # Rank 0 reports epoch statistics
        if rank == 0:
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}: Average loss = {avg_loss:.4f}, Duration = {time.time() - epoch_start:.2f}s")

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed VGG16 training example")
    parser.add_argument("--topk", type=float, default=0.0,
                        help="Fraction of gradient elements to keep for top-k sparsification (e.g. 0.05). Set to 0 to disable.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size-per-gpu", type=int, default=32, help="Per-GPU micro-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate; if unset, use linear scaling heuristic")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights for VGG16 (fine-tuning)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of dataloader worker processes per rank")

    args = parser.parse_args()

    train_vgg16_distributed(
        topk_ratio=args.topk,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_epochs=args.epochs,
        base_lr=args.lr,
        pretrained=args.pretrained,
        num_workers=args.num_workers,
    )
