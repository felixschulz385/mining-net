#!/usr/bin/env bash
#SBATCH --job-name=mining-unet-train
#SBATCH --partition=a100
#SBATCH --qos=gpu1day
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/slurm/mining-unet-train/%j-train.log
#SBATCH --error=logs/slurm/mining-unet-train/%j-train.err

###############################################################################
# SLURM Training Script for Mining UNet
#
# This script trains a UNet model for mining footprint segmentation using:
# - A100 GPU with mixed precision (bfloat16)
# - Zarr backend for efficient data loading
# - Multi-worker data loading (4 workers)
# - Gradient accumulation and periodic cache clearing
#
# Usage:
#   sbatch orchestration/slurm/train_mining_unet.sh [OPTIONS]
#
# Environment:
#   - Set PROJECT_ROOT environment variable or run from project root
#   - Python venv should be at ~/venvs/mn
#   - Data should be in data/landsat_zarr/data.zarr
#
###############################################################################

set -euo pipefail

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

###############################################################################
# Configuration
###############################################################################

# Project paths
PROJECT_ROOT="${PROJECT_ROOT:-.}"
VENV_PATH="${HOME}/venvs/mn"
LOG_DIR="${PROJECT_ROOT}/logs/slurm"
CHECKPOINT_DIR="${PROJECT_ROOT}/models/checkpoints"

# Training parameters (override with command-line args)
COUNTRIES="${1:-}"  # e.g., "ZAF USA"
YEARS="${2:-2019}"
BATCH_SIZE="${3:-64}"
EPOCHS="${4:-100}"
RUN_NAME="${5:-mining_unet_$(date +%Y%m%d_%H%M%S)}"

# Ensure directories exist
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

###############################################################################
# Functions
###############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Mining UNet Training - SLURM Job                             ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_config() {
    echo "Configuration:"
    echo "  Project Root:      ${PROJECT_ROOT}"
    echo "  Venv:              ${VENV_PATH}"
    echo "  Job ID:            ${SLURM_JOB_ID}"
    echo "  Node:              ${SLURM_NODELIST}"
    echo "  CPUs:              ${SLURM_CPUS_PER_TASK}"
    echo ""
    echo "Training Parameters:"
    echo "  Countries:         ${COUNTRIES:-all}"
    echo "  Years:             ${YEARS}"
    echo "  Batch Size:        ${BATCH_SIZE}"
    echo "  Epochs:            ${EPOCHS}"
    echo "  Run Name:          ${RUN_NAME}"
    echo ""
}

###############################################################################
# Setup
###############################################################################

print_header

log_info "Starting training setup..."
print_config

# Change to project directory
cd "${PROJECT_ROOT}"
log_info "Working directory: $(pwd)"

# Load modules
log_info "Loading required modules..."
module purge
module load CUDA/13.0.0
log_success "Modules loaded"

# Activate venv
log_info "Activating Python venv..."
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    log_error "Venv not found at ${VENV_PATH}"
    exit 1
fi
source "${VENV_PATH}/bin/activate"
log_success "Venv activated: $(python --version)"

# Verify environment
log_info "Verifying environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" || log_warning "GPU detection failed"
python -c "import zarr; print(f'Zarr: {zarr.__version__}')"
log_success "Environment verified"

# Check data availability
log_info "Checking data availability..."
DATA_PATH="${PROJECT_ROOT}/data/landsat_zarr/data.zarr"
if [[ ! -d "${DATA_PATH}" ]]; then
    log_error "Data not found at ${DATA_PATH}"
    exit 1
fi
log_success "Data found: ${DATA_PATH}"

###############################################################################
# Training
###############################################################################

log_info "Starting training job..."
echo ""

# Build training command
TRAIN_CMD="python src/network/train.py"
TRAIN_CMD="${TRAIN_CMD} --run-name ${RUN_NAME}"
TRAIN_CMD="${TRAIN_CMD} --epochs ${EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --batch-size ${BATCH_SIZE}"

if [[ -n "${YEARS}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --years ${YEARS}"
fi

if [[ -n "${COUNTRIES}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --countries ${COUNTRIES}"
fi

log_info "Training command:"
echo "  ${TRAIN_CMD}"
echo ""

# Run training with error handling
if ${TRAIN_CMD}; then
    log_success "Training completed successfully"
    TRAIN_EXIT_CODE=0
else
    TRAIN_EXIT_CODE=$?
    log_error "Training failed with exit code ${TRAIN_EXIT_CODE}"
fi

###############################################################################
# Post-Training
###############################################################################

echo ""
log_info "Training job summary:"
echo "  Job ID:            ${SLURM_JOB_ID}"
echo "  Run Name:          ${RUN_NAME}"
echo "  Exit Code:         ${TRAIN_EXIT_CODE}"
echo "  Checkpoints:       ${CHECKPOINT_DIR}/${RUN_NAME}/"
echo "  Logs:              ${PROJECT_ROOT}/logs/${RUN_NAME}/"
echo ""

# Memory usage
log_info "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || log_warning "nvidia-smi not available"

if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    log_success "Job completed successfully!"
else
    log_error "Job failed - check logs at logs/slurm/${SLURM_JOB_ID}-train.err"
fi

echo ""

exit ${TRAIN_EXIT_CODE}
