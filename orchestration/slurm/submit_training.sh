#!/usr/bin/env bash
# filepath: orchestration/slurm/submit_training.sh
#
# Helper script to easily submit training jobs with different configurations
# This script wraps sbatch with common presets
#
# Usage:
#   ./submit_training.sh [PRESET]
#
# Presets:
#   quick     - Quick test run (1 epoch, small batch)
#   medium    - Standard training (100 epochs, batch=64)
#   full      - Full training with early stopping
#   debug     - Debug mode with minimal resources

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_SCRIPT="${PROJECT_ROOT}/orchestration/slurm/train_mining_unet.sh"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
    echo -e "${YELLOW}Error: SLURM script not found at ${SLURM_SCRIPT}${NC}"
    exit 1
fi

# Default preset
PRESET="${1:-medium}"

# Configuration presets
case "${PRESET}" in
    quick)
        echo -e "${BLUE}Quick Test (1 epoch, batch=64)${NC}"
        COUNTRIES=""
        YEARS="2019"
        BATCH_SIZE="64"
        EPOCHS="1"
        TIME_LIMIT="01:00:00"
        CPUS="4"
        MEMORY="32G"
        ;;
    medium)
        echo -e "${BLUE}Standard Training (100 epochs, batch=64)${NC}"
        COUNTRIES=""
        YEARS="2019"
        BATCH_SIZE="64"
        EPOCHS="100"
        TIME_LIMIT="12:00:00"
        CPUS="8"
        MEMORY="64G"
        ;;
    full)
        echo -e "${BLUE}Full Training (all data, 150 epochs)${NC}"
        COUNTRIES=""
        YEARS=""
        BATCH_SIZE="64"
        EPOCHS="150"
        TIME_LIMIT="24:00:00"
        CPUS="8"
        MEMORY="64G"
        ;;
    debug)
        echo -e "${BLUE}Debug Mode (1 epoch, batch=8, verbose)${NC}"
        COUNTRIES=""
        YEARS="2019"
        BATCH_SIZE="8"
        EPOCHS="1"
        TIME_LIMIT="00:30:00"
        CPUS="2"
        MEMORY="16G"
        ;;
    *)
        echo -e "${YELLOW}Unknown preset: ${PRESET}${NC}"
        echo "Available presets: quick, medium (default), full, debug"
        exit 1
        ;;
esac

# Generate run name
RUN_NAME="${PRESET}_$(date +%Y%m%d_%H%M%S)"

# Display configuration
echo ""
echo "Job Configuration:"
echo "  Preset:       ${PRESET}"
echo "  Run Name:     ${RUN_NAME}"
echo "  Epochs:       ${EPOCHS}"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Years:        ${YEARS:-all}"
echo "  Time Limit:   ${TIME_LIMIT}"
echo "  CPUs:         ${CPUS}"
echo "  Memory:       ${MEMORY}"
echo ""

# Confirm before submitting
read -r -p "Submit job? [Y/n] " CONFIRM
CONFIRM=${CONFIRM:-Y}

if [[ ! "${CONFIRM}" =~ ^[Yy] ]]; then
    echo -e "${YELLOW}Cancelled.${NC}"
    exit 0
fi

# Submit job with sbatch
echo -e "${BLUE}Submitting job...${NC}"
JOB_ID=$(sbatch \
    --job-name="mining-train-${PRESET}" \
    --time="${TIME_LIMIT}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEMORY}" \
    "${SLURM_SCRIPT}" \
    "${COUNTRIES}" "${YEARS}" "${BATCH_SIZE}" "${EPOCHS}" "${RUN_NAME}" \
    | awk '{print $4}')

echo -e "${GREEN}Job submitted successfully!${NC}"
echo ""
echo "Job ID:     ${JOB_ID}"
echo "Run Name:   ${RUN_NAME}"
echo ""
echo "Monitor job with:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f logs/slurm/${JOB_ID}-train.log"
echo ""
echo "Cancel job with:"
echo "  scancel ${JOB_ID}"
