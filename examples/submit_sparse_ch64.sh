SPARSITY=.99      # 0.0 = baseline, 0.99 = 99% pruning
SAMPLE_PCT=0.01   # % of grad elements sampled for threshold (100=exact, lower=faster/approx)
NCHANNELS=32  # pinned channel count for this sweep point

export USE_SPARSE_RS=0
export USE_SPARSE_AR=1
export AXONN_PRUNE_RS=0
export AXONN_PRUNE_AR=1
export AXONN_PRUNE_SPARSITY=$SPARSITY
export AXONN_PRUNE_SAMPLE_PCT=$SAMPLE_PCT
export SPARSE_COMMS_LOG_SPARSITY=0
export NCCL_RS_SHIM_TIMING=0
export NCCL_RS_SHIM_STATS=0

# export 
# use ncclx 
export LD_PRELOAD="/pscratch/sd/e/egencer/sparsecomms/torchcomms-sparse/build/ncclx/lib/libnccl.so.2"
# use stock nccl with shim to add symbols for sparse collectives
# export LD_PRELOAD="/pscratch/sd/e/egencer/sparsecomms/Megatron-AxoNN/libnccl_sparse_stub.so"

# export LD_PRELOAD="/pscratch/sd/e/egencer/sparsecomms/Megatron-AxoNN/libnccl_sparse_stub.so"
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