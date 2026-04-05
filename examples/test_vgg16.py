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
    ``--log-grad-messages``; the logging uses AxoNN's data-parallel group
    and will be skipped if AxoNN is not initialized.
"""

import os
import time

import argparse

import torch
from typing import Optional
import torch.distributed as dist
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.models import VGG16_Weights
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from axonn import axonn as ax
from axonn.trition_pruner import TritonGradientPruner

_SPARSE_COMMS = None


def _resolve_sparse_collective_modes(use_sparse_collectives: Optional[bool] = None):
    """Resolve per-op sparse collective modes from CLI intent or environment."""
    if use_sparse_collectives is True:
        return {"USE_SPARSE_RS": True, "USE_SPARSE_AR": True, "USE_SPARSE_AG": True}
    if use_sparse_collectives is False:
        return {"USE_SPARSE_RS": False, "USE_SPARSE_AR": False, "USE_SPARSE_AG": False}

    return {
        "USE_SPARSE_RS": os.environ.get("USE_SPARSE_RS", "0") == "1",
        "USE_SPARSE_AR": os.environ.get("USE_SPARSE_AR", "0") == "1",
        "USE_SPARSE_AG": os.environ.get("USE_SPARSE_AG", "0") == "1",
    }


def _load_sparse_comms():
    """Load sparse collective bindings."""
    global _SPARSE_COMMS
    if _SPARSE_COMMS is None:
        from axonn import sparse_comms as sparse_comms_mod

        _SPARSE_COMMS = sparse_comms_mod
    return _SPARSE_COMMS


def _normalize_sample_rate_pct(value: float) -> float:
    """Normalize a user-supplied sample rate to a percentage value."""
    if value <= 1.0:
        return value * 100.0
    return value


def _resolve_prune_sample_pct(prune_sample_pct: Optional[float]) -> tuple[float, str]:
    """Resolve the pruning sample rate as a percentage and report where it came from."""
    sample_pct_env = os.environ.get("AXONN_PRUNE_SAMPLE_PCT")
    if sample_pct_env is not None:
        try:
            return _normalize_sample_rate_pct(float(sample_pct_env)), "env:AXONN_PRUNE_SAMPLE_PCT"
        except Exception:
            pass

    if prune_sample_pct is not None:
        return _normalize_sample_rate_pct(float(prune_sample_pct)), "cli:--prune-sample-pct"

    return 10.0, "default:10.0"


def _cuda_timing_event_pair():
    """Create a CUDA start/end event pair for elapsed-time measurement."""
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _collect_data_parallel_grads(model):
    """Return (name, parameter, grad) tuples reduced across AxoNN data-parallel group."""
    grads = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        grad = getattr(p, "grad", None)
        if grad is None:
            continue

        # Mirrors AxoNN's sync_gradients contract for tensor-parallel params.
        if hasattr(p, "is_tensor_parallel") and p.is_tensor_parallel:
            if not hasattr(p, "needs_depth_parallel_gradient_sync"):
                raise ValueError(
                    f"Tensor-parallel parameter '{name}' is missing needs_depth_parallel_gradient_sync"
                )

        grads.append((name, p, grad))
    return grads


@torch.no_grad()
def sync_gradients_data_parallel(
    model,
    sparse_comms_mod,
    use_sparse: bool,
    mean: bool = True,
    gradient_pruner=None,
):
    """Synchronize gradients across AxoNN data-parallel ranks using sparse AR or dense collectives.

    When `use_sparse` is True and `sparse_comms_mod` is available this uses
    the sparse binder's `all_reduce_sparse`. Otherwise it falls back to
    `torch.distributed.all_reduce` on AxoNN's data-parallel group.
    """
    if not (dist.is_initialized() and hasattr(ax, "comm_handle") and getattr(ax.comm_handle, "data_parallel_group", None) is not None):
        return

    data_parallel_group = ax.comm_handle.data_parallel_group
    group_size = ax.comm_handle.G_data
    grads = _collect_data_parallel_grads(model)
    stats = {
        "prune_sample_ms": 0.0,
        "prune_threshold_ms": 0.0,
        "prune_kernel_ms": 0.0,
    }

    if use_sparse and sparse_comms_mod is not None:
        # Bucket gradients to reduce number of prune kernels and collectives.
        # Buckets group tensors by device and dtype and aim for ~BUCKET_BYTES per bucket.
        BUCKET_BYTES = int(os.environ.get("AXONN_GRAD_BUCKET_BYTES", str(8 * 1024 * 1024)))
        buckets = []
        cur_bucket = []
        cur_bytes = 0

        def flush_bucket():
            nonlocal cur_bucket, cur_bytes
            if cur_bucket:
                buckets.append(cur_bucket)
            cur_bucket = []
            cur_bytes = 0

        # Build buckets (keep device/dtype homogeneous within a bucket)
        first_device = None
        first_dtype = None
        for name, param, grad in grads:
            if grad is None:
                continue
            elem_bytes = int(grad.element_size() * grad.numel())
            if not cur_bucket:
                cur_bucket.append((name, param, grad))
                cur_bytes = elem_bytes
                first_device = grad.device
                first_dtype = grad.dtype
                continue

            if grad.device != first_device or grad.dtype != first_dtype or (cur_bytes + elem_bytes) > BUCKET_BYTES:
                flush_bucket()
                first_device = grad.device
                first_dtype = grad.dtype
                cur_bucket.append((name, param, grad))
                cur_bytes = elem_bytes
            else:
                cur_bucket.append((name, param, grad))
                cur_bytes += elem_bytes

        flush_bucket()

        handles = []
        # Process each bucket: concat -> prune -> all_reduce_sparse -> scatter back
        for bucket in buckets:
            views = [g.reshape(-1) for (_, _, g) in bucket]
            if not views:
                continue
            bucket_flat = torch.cat(views, dim=0)

            # Record pre-collective nonzero/total and per-bucket stats
            try:
                nz = int(torch.count_nonzero(bucket_flat).item())
                tot = int(bucket_flat.numel())
                stats.setdefault("precollective_nonzero", 0)
                stats.setdefault("precollective_total", 0)
                stats["precollective_nonzero"] += nz
                stats["precollective_total"] += tot
                stats.setdefault("buckets", [])
                stats["buckets"].append({"bytes": int(bucket_flat.element_size() * bucket_flat.numel()), "nonzero": nz, "total": tot})
            except Exception:
                pass

            # Create a stable bucket key from parameter pointers so error-feedback persists
            bucket_key = tuple(int(p.data_ptr()) for (_, p, _) in bucket)

            if gradient_pruner is not None:
                _, prune_timing = gradient_pruner.prune(bucket_flat, key=bucket_key, return_timing=True)
                stats.setdefault("prune_timing_events", [])
                stats["prune_timing_events"].append(prune_timing)

            # Launch sparse all-reduce on the bucket buffer (in-place)
            handle = sparse_comms_mod.all_reduce_sparse(
                bucket_flat,
                group=data_parallel_group,
                async_op=True,
            )
            if handle is not None:
                handles.append((handle, bucket))

        # Wait for all collectives and scatter results back into original grads
        for handle, bucket in handles:
            handle.wait()
            # Reconstruct concatenated buffer from the (now reduced) per-param grads
            views = [g.reshape(-1) for (_, _, g) in bucket]
            if not views:
                continue
            bucket_flat_after = torch.cat(views, dim=0)
            offset = 0
            for (_, param, grad) in bucket:
                n = grad.numel()
                grad.copy_(bucket_flat_after[offset : offset + n].view_as(grad))
                offset += n
    else:
        # Dense path: enqueue async all-reduces and join once.
        handles = []
        for _, _, grad in grads:
            handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=data_parallel_group, async_op=True)
            if handle is not None:
                handles.append(handle)
        for handle in handles:
            handle.wait()

    if mean and group_size > 1:
        scale = 1.0 / float(group_size)
        for _, _, grad in grads:
            grad.mul_(scale)

    return stats

def log_grad_message_sizes(model, sparse_comms_mod, use_sparse: bool) -> None:
    """Compute and report per-data-parallel-rank gradient-message sizes.

    This computes the approximate number of bytes that would be communicated
    during an all-reduce of gradients for ``model``. The function requires
    AxoNN's data-parallel group (``ax.comm_handle.data_parallel_group``)
    and will skip logging if AxoNN is not initialized.
    """
    # Classify gradients the same way AxoNN's `sync_gradients` does, so
    # we count only those gradients that will be reduced over the
    # data-parallel group.
    if not (dist.is_initialized() and hasattr(ax, "comm_handle") and getattr(ax.comm_handle, "data_parallel_group", None) is not None):
        if dist.is_initialized() and dist.get_rank() == 0:
            print("AxoNN data-parallel group unavailable — skipping gradient-size logging")
        return

    data_parallel_group = ax.comm_handle.data_parallel_group
    group_size = ax.comm_handle.G_data
    rank_in_group = ax.comm_handle.data_parallel_rank

    grads = _collect_data_parallel_grads(model)

    def bytes_of(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    local_total = sum(bytes_of(grad) for _, _, grad in grads)

    # Gather per-group-rank totals using AxoNN's data-parallel group
    try:
        device = next(model.parameters()).device
        tensor_device = device if device.type == "cuda" else torch.device("cpu")
    except StopIteration:
        tensor_device = torch.device("cpu")

    if not use_sparse or sparse_comms_mod is None:
        if rank_in_group == 0:
            print("Sparse collectives unavailable — skipping gradient-size logging")
        return

    local = torch.tensor([local_total], dtype=torch.long, device=tensor_device)
    # use binder's all_gather_sparse when available
    gathered = torch.empty(group_size, dtype=torch.long, device=tensor_device)
    sparse_comms_mod.all_gather_sparse(local, gathered, group=data_parallel_group, async_op=False)
    gathered_ints = [int(x.item()) for x in gathered]

    if rank_in_group == 0:
        print("Per-data-parallel-rank all-reduce byte counts (approx):")
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
                            prune_sample_pct=None,
                            batch_size_per_gpu=32,
                            num_epochs=30,
                            base_lr=None,
                            pretrained=False,
                            dataset_root=None,
                            split="train",
                            num_classes=256,
                            num_workers=None,
                            optimizer_name="adamw",
                            log_grad_messages: bool = False,
                            use_sparse_collectives: Optional[bool] = None):
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

    resolved_prune_sample_pct, resolved_prune_sample_pct_source = _resolve_prune_sample_pct(prune_sample_pct)

    gradient_pruner = None
    if topk_ratio is not None and 0.0 < topk_ratio < 1.0:
        # Determine sample_pct for approximate thresholding.
        sample_pct = resolved_prune_sample_pct

        sparsity_env = os.environ.get("AXONN_PRUNE_SPARSITY")
        try:
            sparsity = float(sparsity_env) if sparsity_env is not None else 1.0 - topk_ratio
        except Exception:
            sparsity = 1.0 - topk_ratio

        gradient_pruner = TritonGradientPruner(sparsity=sparsity, sample_pct=sample_pct)

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    ax.init(G_data=num_gpus, G_inter=1)

    # Determine whether to use sparse collectives. Dense is the default unless
    # the caller explicitly enables sparse or the environment requests it.
    sparse_modes = _resolve_sparse_collective_modes(use_sparse_collectives)
    use_sparse = any(sparse_modes.values())

    sparse_comms_mod = None
    if use_sparse:
        try:
            sparse_comms_mod = _load_sparse_comms()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load axonn.sparse_comms. Ensure NCCLX/NCCL build paths are set "
                "(e.g., NCCLX_BUILD_DIR) before running distributed training."
            ) from exc

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Initialized distributed training on {world_size} processes (GPUs)")
        print(f"Global batch size: {global_batch_size}")
        print(f"Base LR: {base_lr:.6f}")
        # Log key CLI arguments and environment variables used by this run
        print(
            f"  topk={topk_ratio}, prune_sample_pct={resolved_prune_sample_pct:.6f} "
            f"(source={resolved_prune_sample_pct_source}), batch_size_per_gpu={batch_size_per_gpu}, optimizer={optimizer_name}"
        )
        print(f"  Resolved collectives mode (CLI/env): use_sparse={use_sparse}")
        print(
            "  Resolved per-op sparse modes: "
            f"RS={sparse_modes['USE_SPARSE_RS']}, "
            f"AR={sparse_modes['USE_SPARSE_AR']}, "
            f"AG={sparse_modes['USE_SPARSE_AG']}"
        )
        print(f"  ENV: AXONN_PRUNE_SPARSITY={os.environ.get('AXONN_PRUNE_SPARSITY')}, AXONN_PRUNE_SAMPLE_PCT={os.environ.get('AXONN_PRUNE_SAMPLE_PCT')}")
        print(f"  ENV: USE_SPARSE_AR={os.environ.get('USE_SPARSE_AR')}, USE_SPARSE_AG={os.environ.get('USE_SPARSE_AG')}")
        print(f"  ENV: NCCLX_BUILD_DIR={os.environ.get('NCCLX_BUILD_DIR')}, NCCL_HOME={os.environ.get('NCCL_HOME')}, LD_PRELOAD={os.environ.get('LD_PRELOAD')}")

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
        # Ensure images are RGB to avoid grayscale images from some datasets
        transforms.Lambda(lambda img: img.convert("RGB") if hasattr(img, "convert") else img),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dist.barrier(device_ids=[torch.cuda.current_device()])

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

    run_prune_time_ms = 0.0
    run_prune_sample_time_ms = 0.0
    run_prune_threshold_time_ms = 0.0
    run_prune_kernel_time_ms = 0.0
    run_allreduce_time_ms = 0.0
    run_compute_time_ms = 0.0

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        epoch_prune_time_ms = 0.0
        epoch_prune_sample_time_ms = 0.0
        epoch_prune_threshold_time_ms = 0.0
        epoch_prune_kernel_time_ms = 0.0
        epoch_allreduce_time_ms = 0.0
        epoch_compute_time_ms = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (x, y) in enumerate(tqdm(train_dataloader, disable=(rank != 0), desc=f"Epoch {epoch+1}/{num_epochs}")):
            x, y = x.cuda(), y.cuda()

            compute_start, compute_end = _cuda_timing_event_pair()
            compute_start.record()

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            # Optionally log approximate gradient-message sizes (aggregated by AxoNN).
            # Suppress logging errors so that training is not interrupted.
            if log_grad_messages:
                try:
                    log_grad_message_sizes(model, sparse_comms_mod, use_sparse=use_sparse)
                except Exception:
                    pass

            # compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            sync_start, sync_end = _cuda_timing_event_pair()
            sync_start.record()
            sync_stats = sync_gradients_data_parallel(
                model,
                sparse_comms_mod,
                use_sparse=use_sparse,
                mean=True,
                gradient_pruner=gradient_pruner,
            )
            if sync_stats is None:
                sync_stats = {
                    "prune_sample_ms": 0.0,
                    "prune_threshold_ms": 0.0,
                    "prune_kernel_ms": 0.0,
                }
            sync_end.record()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            compute_end.record()
            compute_end.synchronize()

            # If the sparse path returned Triton timing events, convert them
            # to numeric milliseconds now that we've synchronized the GPU.
            if sync_stats is not None and "prune_timing_events" in sync_stats:
                # Ensure numeric keys exist
                sync_stats.setdefault("prune_sample_ms", 0.0)
                sync_stats.setdefault("prune_threshold_ms", 0.0)
                sync_stats.setdefault("prune_kernel_ms", 0.0)
                for timing in sync_stats.get("prune_timing_events", []):
                    try:
                        s_ms = float(timing["sample_start"].elapsed_time(timing["sample_end"]))
                        t_ms = float(timing["threshold_start"].elapsed_time(timing["threshold_end"]))
                        k_ms = float(timing["prune_start"].elapsed_time(timing["prune_end"]))
                    except RuntimeError:
                        # Event pairs not completed yet; skip this timing entry
                        continue
                    except Exception:
                        # Unexpected error — raise to avoid silently masking issues
                        raise
                    sync_stats["prune_sample_ms"] += s_ms
                    sync_stats["prune_threshold_ms"] += t_ms
                    sync_stats["prune_kernel_ms"] += k_ms

            compute_time_ms = compute_start.elapsed_time(compute_end)
            sync_time_ms = sync_start.elapsed_time(sync_end)

            prune_sample_time_ms = float(sync_stats["prune_sample_ms"])
            prune_threshold_time_ms = float(sync_stats["prune_threshold_ms"])
            prune_kernel_time_ms = float(sync_stats["prune_kernel_ms"])

            prune_time_ms = prune_sample_time_ms + prune_threshold_time_ms + prune_kernel_time_ms
            allreduce_time_ms = max(0.0, sync_time_ms - prune_time_ms)
            compute_rest_time_ms = max(0.0, compute_time_ms - prune_time_ms - allreduce_time_ms)

            run_compute_time_ms += compute_time_ms
            run_prune_time_ms += prune_time_ms
            run_prune_sample_time_ms += prune_sample_time_ms
            run_prune_threshold_time_ms += prune_threshold_time_ms
            run_prune_kernel_time_ms += prune_kernel_time_ms
            run_allreduce_time_ms += allreduce_time_ms
            epoch_compute_time_ms += compute_time_ms
            epoch_prune_time_ms += prune_time_ms
            epoch_prune_sample_time_ms += prune_sample_time_ms
            epoch_prune_threshold_time_ms += prune_threshold_time_ms
            epoch_prune_kernel_time_ms += prune_kernel_time_ms
            epoch_allreduce_time_ms += allreduce_time_ms

            batch_loss = float(loss.item())
            epoch_loss += batch_loss

            # training accuracy accumulation
            preds = logits.argmax(dim=1)
            train_correct += int((preds == y).sum().item())
            train_total += x.size(0)

            # Print per-batch statistics (only on rank 0).
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx+1}: loss = {batch_loss:.6f} | LR = {current_lr:.6e} | grad_norm = {total_norm:.6f}")
                print(f"    Timing (ms, CUDA events): allreduce={allreduce_time_ms:.6f}, prune={prune_time_ms:.6f}, compute={compute_rest_time_ms:.6f}")
                # If requested, print pre-collective gradient-message sizes and bucket sparsity
                if log_grad_messages:
                    try:
                        pre_nz = sync_stats.get("precollective_nonzero") if sync_stats is not None else None
                        pre_tot = sync_stats.get("precollective_total") if sync_stats is not None else None
                        if pre_nz is not None and pre_tot is not None:
                            sparsity_pct = 100.0 * (1.0 - float(pre_nz) / float(max(1, pre_tot)))
                            print(f"    Pre-collective nonzeros: {pre_nz}/{pre_tot} (sparsity ~ {sparsity_pct:.2f}%)")
                        buckets = sync_stats.get("buckets") if sync_stats is not None else None
                        if buckets:
                            # Print a short summary: count, avg bucket bytes, avg sparsity
                            cnt = len(buckets)
                            avg_bytes = sum(b["bytes"] for b in buckets) / float(cnt)
                            avg_nz = sum(b["nonzero"] for b in buckets) / float(cnt)
                            avg_tot = sum(b["total"] for b in buckets) / float(cnt)
                            avg_sparsity = 100.0 * (1.0 - (avg_nz / max(1.0, avg_tot)))
                            print(f"    Buckets: {cnt}, avg_size={avg_bytes/1024.0:.1f}KB, avg_sparsity~{avg_sparsity:.2f}%")
                    except Exception:
                        pass

        if rank == 0:
            avg_loss = epoch_loss / max(1, len(train_dataloader))
            train_acc = 100.0 * train_correct / max(1, train_total)
            print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}, Train Acc = {train_acc:.2f}%, Duration = {time.time()-epoch_start:.2f}s")
            print(f"Epoch timing summary (ms, CUDA events): allreduce_total={epoch_allreduce_time_ms:.6f}, prune_total={epoch_prune_time_ms:.6f}, compute_total={max(0.0, epoch_compute_time_ms-epoch_prune_time_ms-epoch_allreduce_time_ms):.6f}")
            if gradient_pruner is not None:
                print(
                    f"Epoch prune breakdown (ms, CUDA events): sample_total={epoch_prune_sample_time_ms:.6f}, "
                    f"threshold_total={epoch_prune_threshold_time_ms:.6f}, kernel_total={epoch_prune_kernel_time_ms:.6f}"
                )

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
            # Skip validation if the 'val' split is unavailable or loader raises an error.
            if rank == 0:
                print("Validation skipped (no 'val' split or loader error)")

    if rank == 0:
        print(
            "Run timing summary (ms, CUDA events): "
            f"allreduce_total={run_allreduce_time_ms:.6f}, "
            f"prune_total={run_prune_time_ms:.6f}, "
            f"compute_total={max(0.0, run_compute_time_ms-run_prune_time_ms-run_allreduce_time_ms):.6f}"
        )
        if gradient_pruner is not None:
            print(
                "Run prune breakdown (ms, CUDA events): "
                f"sample_total={run_prune_sample_time_ms:.6f}, "
                f"threshold_total={run_prune_threshold_time_ms:.6f}, "
                f"kernel_total={run_prune_kernel_time_ms:.6f}"
            )

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train VGG16 on Caltech-256 with AxoNN")
    parser.add_argument("--topk", type=float, default=0.0, help="Fraction of gradients to keep via top-k sparsification (0 disables)")
    parser.add_argument("--prune-sample-pct", type=float, default=None, help="Sample rate for TritonGradientPruner threshold estimation; values <=1 are treated as fractions (0.01 -> 1%). Overrides AXONN_PRUNE_SAMPLE_PCT env var")
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
    parser.add_argument("--log-grad-messages", action="store_true", dest="log_grad_messages", help="Enable aggregated gradient-message-size logging (prints per-data-parallel-rank totals on AxoNN group-rank 0)")
    parser.add_argument("--sparse-collectives", action="store_true", dest="sparse_collectives", help="Use sparse collectives (overrides env vars)")
    parser.add_argument("--dense-collectives", action="store_true", dest="dense_collectives", help="Use dense collectives (overrides env vars)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a CPU-only dry-run: validate dataset and model instantiation without distributed init or GPUs")

    args = parser.parse_args()
    if args.sparse_collectives and args.dense_collectives:
        parser.error("--sparse-collectives and --dense-collectives are mutually exclusive")

    requested_modes = _resolve_sparse_collective_modes(
        True if args.sparse_collectives else False if args.dense_collectives else None,
    )
    for env_name, enabled in requested_modes.items():
        os.environ[env_name] = "1" if enabled else "0"

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
        # Determine collectives mode: CLI overrides env defaults from submit script.
        if args.sparse_collectives:
            use_sparse = True
            topk_ratio = args.topk
        elif args.dense_collectives:
            use_sparse = False
            topk_ratio = 0.0
        else:
            # follow the resolved environment defaults: dense unless sparse was explicitly enabled
            use_sparse = any(requested_modes.values())
            topk_ratio = args.topk

        train_vgg16_distributed(
            topk_ratio=topk_ratio,
            prune_sample_pct=args.prune_sample_pct,
            batch_size_per_gpu=args.batch_size_per_gpu,
            num_epochs=args.epochs,
            base_lr=args.lr,
            pretrained=args.pretrained,
            dataset_root=args.dataset_root,
            split=args.split,
            num_classes=args.num_classes,
            num_workers=args.num_workers,
            optimizer_name=args.optimizer,
            log_grad_messages=args.log_grad_messages,
            use_sparse_collectives=use_sparse,
        )


if __name__ == "__main__":
    main()
