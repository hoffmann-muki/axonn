#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint="gpu\&hbm40g"
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --account=m5083_g
#SBATCH --job-name=sparse_ch32
#SBATCH --output=logs/sparse_ch32_%j.out
#SBATCH --error=logs/sparse_ch32_%j.err

set -euo pipefail

# activate conda env for torchcomms
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    . "$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

conda activate torchcomms

SPARSITY=.99      # 0.0 = baseline, 0.99 = 99% pruning
SAMPLE_PCT=0.01   # % of grad elements sampled for threshold (100=exact, lower=faster/approx)
NCHANNELS=32  # pinned channel count for this sweep point

# Ensure conda environment and NCCL/conda paths are set before other vars
export NCCL_HOME=/pscratch/sd/h/hmuki/torchcomms-sparse/build/ncclx
export CONDA_PREFIX=/pscratch/sd/h/hmuki/miniconda3/envs/torchcomms
export CONDA_LIB_DIR=$CONDA_PREFIX/lib
export CONDA_INCLUDE_DIR=$CONDA_PREFIX/include
export NCCLX_BUILD_DIR="$NCCL_HOME"

# Make torchrun discoverable from both the env and the base conda installation.
export PATH="$CONDA_PREFIX/bin:$CONDA_BASE/bin:$PATH"
TORCHRUN_BIN=$(command -v torchrun)
if [ -z "$TORCHRUN_BIN" ]; then
    echo "torchrun is not available on PATH after conda activation" >&2
    exit 1
fi

# Ensure runtime linker can find NCCL, conda, and system libs (e.g., system OpenSSL for NCCLX)
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH:$CONDA_LIB_DIR

export USE_SPARSE_RS=0
export USE_SPARSE_AR=1
export AXONN_PRUNE_RS=0
export AXONN_PRUNE_AR=1
export AXONN_PRUNE_SPARSITY=$SPARSITY
export AXONN_PRUNE_SAMPLE_PCT=$SAMPLE_PCT
export SPARSE_COMMS_LOG_SPARSITY=0
export NCCL_RS_SHIM_TIMING=0
export NCCL_RS_SHIM_STATS=0
 
# use ncclx
export LD_PRELOAD="$NCCL_HOME/lib/libnccl.so.2"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
# unset SLURM_MPI_TYPE
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export FI_PROVIDER=cxi
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_OPTIMIZED_MRS=0
# export FI_CXI_DISABLE_HMEM_DEV_REGISTER=1
# export FI_CXI_OFLOW_BUF_SIZE=1073741824
# export FI_CXI_OFLOW_BUF_COUNT=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_ALLREDUCE_USE_KERNEL=1
export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:3; 1:2; 2:1; 3:0"
# --- CCD sparse collective flags (adaptive_spop, FORMAT_MASK=5) ---
export NCCL_BUFFSIZE=4404032
# export NCCL_BUFFSIZE=8388608
# export NCCL_BUFFSIZE=16777216
export NCCL_CCD_FORMAT_MASK=5
export NCCL_CCD_DENSE_THRESHOLD=0.6
export NCCL_CCD_DENSE_INTRA_THRESHOLD=0.7
export NCCL_CCD_CHANNELS=$NCHANNELS
# export NCCL_MIN_NCHANNELS=$NCHANNELS
export NCCL_MAX_NCHANNELS=$NCHANNELS

# Move to working directory on shared scratch
cd /pscratch/sd/h/hmuki/axonn

# Create logs directory in the working directory
mkdir -p logs

# Rendezvous and torch cache settings
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export TORCH_HOME=/pscratch/sd/h/hmuki/.cache_general/torch

# Ensure sparse_comms build cache directory exists and is writable
export SPARSE_COMMS_BUILD_DIR=/pscratch/sd/h/hmuki/sparse_comms_build
mkdir -p "$SPARSE_COMMS_BUILD_DIR"

# Run distributed training with torchrun once it has been made discoverable on PATH
srun "$TORCHRUN_BIN" \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  examples/test_vgg16.py \
  --dataset-root /pscratch/sd/h/hmuki/axonn/caltech256 \
  --split train \
  --batch-size-per-gpu 2 \
  --epochs 2 \
  --sparse-collectives \
  --pretrained