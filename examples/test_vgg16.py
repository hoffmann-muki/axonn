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
import torchvision
import torchvision.models as models
from torchvision.transforms import ToTensor
from tqdm import tqdm

from axonn import axonn as ax

from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

@torch.no_grad()
def apply_topk_sparsification(model, ratio=0.05):
    """
    Sparsifies gradients by keeping only the top-k magnitude values
    and zeroing out the rest.
    """
    for p in model.parameters():
        if p.grad is None:
            continue
            
        # Flatten the gradient to 1D
        grad_flat = p.grad.view(-1)
        total_elements = grad_flat.numel()
        
        # Calculate k (e.g., 5% of total elements)
        k = max(1, int(total_elements * ratio))
        
        # Get values and indices of Top-K magnitudes
        # We use .abs() because a large negative gradient is just as important as a positive one
        _, indices = torch.topk(grad_flat.abs(), k)
        
        # Create a dense mask or zero-filled tensor
        # Then scatter the original values back into the top-k positions
        topk_values = grad_flat[indices]
        new_grad = torch.zeros_like(grad_flat)
        new_grad.scatter_(0, indices, topk_values)
        
        # Replace the original gradient with the sparsified dense version
        p.grad.copy_(new_grad.view(p.grad.shape))

def train_vgg16_distributed():
    """
    Distributed training routine for VGG16 on synthetic ImageNet data.
    
    Requires:
      - WORLD_SIZE environment variable set to the number of processes
      - Launch via torchrun or mpirun with proper rank environment variables
    """
    # Configuration
    batch_size_per_gpu = 64
    num_gpus = int(os.environ["WORLD_SIZE"])
    global_batch_size = num_gpus * batch_size_per_gpu
    num_epochs = 10
    learning_rate = 1e-3

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
    model = models.vgg16().cuda()
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-6)

    # Learning rate scheduler with warmup
    # We ramp up for 1 epoch then decay
    warmup_steps = 18 
    total_steps = 18 * num_epochs
    
    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    # Construct distributed dataloader
    # ax.create_dataloader handles data sharding across data-parallel ranks
    train_dataset = torchvision.datasets.FakeData(
        size=64 * 12 * 12, # 9216 samples / 512 global batch = 18 iterations per epoch
        image_size=(3, 224, 224),
        num_classes=1000,
        transform=ToTensor(),
    )
    train_dataloader = ax.create_dataloader(
        dataset=train_dataset,
        global_batch_size=global_batch_size,
        micro_batch_size=batch_size_per_gpu,
        num_workers=0,
    )

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
            
            apply_topk_sparsification(model)

            # Gradient clipping to help with stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
    train_vgg16_distributed()
