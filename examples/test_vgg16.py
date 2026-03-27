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
    num_epochs = 2
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
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Construct distributed dataloader
    # ax.create_dataloader handles data sharding across data-parallel ranks
    train_dataset = torchvision.datasets.FakeData(
        size=64 * 12 * 12,
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

        for (x, y) in tqdm(
            train_dataloader,
            disable=(rank != 0),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        ):
            # Move data to GPU
            x, y = x.cuda(), y.cuda()

            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)

            # Backward pass (gradient synchronization occurs automatically via NCCL)
            loss.backward()

            # Parameter update
            optimizer.step()

            epoch_loss += loss.item()

        # Rank 0 reports epoch statistics
        if rank == 0:
            avg_loss = epoch_loss / len(train_dataloader)
            epoch_duration = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}: "
                f"Average loss = {avg_loss:.4f}, "
                f"Duration = {epoch_duration:.2f}s"
            )

    if rank == 0:
        print("Training procedure completed.")




if __name__ == "__main__":
    train_vgg16_distributed()
