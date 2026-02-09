# SLURM Training Scripts

This directory contains SLURM job submission scripts for training the mining UNet model on sciCORE.

## Quick Start

### Easy Submission (Recommended)
```bash
# Make scripts executable
chmod +x orchestration/slurm/*.sh

# Submit with preset configuration
./orchestration/slurm/submit_training.sh [PRESET]
```

### Available Presets
- **quick** - Test run (1 epoch, ~5 mins)
- **medium** - Standard training (100 epochs, ~24 hours)
- **full** - Full training (150 epochs, all data)
- **debug** - Debug mode (1 epoch, minimal resources)

### Examples
```bash
# Quick test
./orchestration/slurm/submit_training.sh quick

# Standard training (default)
./orchestration/slurm/submit_training.sh

# Full training with all data
./orchestration/slurm/submit_training.sh full
```

## Manual Submission

For custom configurations, submit directly with `sbatch`:

```bash
# Minimal training
sbatch orchestration/slurm/train_mining_unet.sh

# Training for specific countries and years
sbatch orchestration/slurm/train_mining_unet.sh "ZAF USA" 2019 64 100

# Custom batch size and epochs
sbatch orchestration/slurm/train_mining_unet.sh "" "2019 2020" 32 200
```

### Command-Line Arguments

```
./train_mining_unet.sh [COUNTRIES] [YEARS] [BATCH_SIZE] [EPOCHS] [RUN_NAME]
```

| Argument | Default | Example |
|----------|---------|---------|
| COUNTRIES | (all) | `ZAF USA` |
| YEARS | 2019 | `2019 2020` |
| BATCH_SIZE | 64 | `32` |
| EPOCHS | 100 | `200` |
| RUN_NAME | timestamp | `my_experiment_v1` |

## Resource Configuration

The base SLURM script requests:
- **Partition**: `a100` (A100 40GB GPU)
- **GPU**: 1x A100
- **CPUs**: 8
- **Memory**: 64GB
- **Time**: 24 hours

To modify resources, edit the SBATCH directives in `train_mining_unet.sh`:

```bash
#SBATCH --partition=a100      # or a100-80g, rtx4090, titan, l40s, h200
#SBATCH --time=24:00:00       # Job time limit
#SBATCH --cpus-per-task=8     # CPU workers for data loading
#SBATCH --mem=64G             # Total memory
```

## Monitoring Training

### Check Job Status
```bash
# View all your jobs
squeue -u schulz0022

# View specific job
squeue -j <JOB_ID>

# Watch job in real-time
watch -n 1 squeue -j <JOB_ID>
```

### Monitor Logs
```bash
# Real-time log streaming
tail -f logs/slurm/<JOB_ID>-train.log

# View errors
tail -f logs/slurm/<JOB_ID>-train.err

# Search for specific patterns
grep "Epoch" logs/slurm/<JOB_ID>-train.log | tail -20
```

### GPU Memory Usage
```bash
# Check during training
ssh <compute_node> nvidia-smi

# Or query from login node
srun -w <compute_node> nvidia-smi
```

## Tensorboard Monitoring

Training metrics are logged to `logs/<RUN_NAME>/`:

```bash
# Start tensorboard (local machine)
tensorboard --logdir=logs

# Or on remote (from VS Code)
python -m tensorboard.main --logdir=/path/to/mining-net/logs
```

## Checkpoints

Model checkpoints are saved to:
```
models/checkpoints/<RUN_NAME>/
├── epoch_001.pth          # Checkpoint every epoch
├── epoch_002.pth
├── best_model.pth         # Best validation loss
└── training_history.png   # Training curves
```

To resume training from checkpoint:
```bash
# Edit train.py to load checkpoint before training
# Then resubmit job with same RUN_NAME
```

## Common Issues

### Job Fails Immediately
- Check logs: `cat logs/slurm/<JOB_ID>-train.err`
- Verify venv exists: `ls ~/venvs/mn/bin/activate`
- Check data: `ls data/landsat_zarr/data.zarr`

### Out of Memory (OOM)
1. Reduce batch size: `--batch-size 32`
2. Increase gradient accumulation in config
3. Reduce model size (FILTERS_BASE, DEPTH)
4. Request A100 80GB: `--partition=a100-80g`

### Data Loading Too Slow
- Check NUM_WORKERS in config (default: 4)
- Check if using persistent_workers (should be True)
- Monitor with: `nvidia-smi | grep python`

### Job Timeout
- Increase time limit: `#SBATCH --time=36:00:00`
- Check max time limits: `sinfo --format="%25N %10c %10m %25f %10T %5D %10t"`

## Performance Tuning

### For Maximum Speed
```python
# In src/network/config.py
BATCH_SIZE = 64              # Larger batches
NUM_WORKERS = 4              # Parallel data loading
PREFETCH_FACTOR = 2          # Buffer prefetch
PERSISTENT_WORKERS = True    # Keep workers alive
EMPTY_CACHE_EVERY_N_STEPS = 100  # Periodic cleanup
USE_AMP = True               # Mixed precision
AMP_DTYPE = 'bfloat16'       # A100 native precision
```

### For Memory Efficiency
```python
BATCH_SIZE = 32              # Smaller batches
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 64
DROPOUT_RATE = 0.3           # More regularization
FILTERS_BASE = 64            # Smaller model
```

## Example Workflows

### Hyperparameter Tuning
```bash
# Test different learning rates
for lr in 1e-4 1e-3 1e-2; do
    ./orchestration/slurm/submit_training.sh "" 2019 64 50 "lr_${lr}"
done
```

### Multi-Year Training
```bash
# Train on multiple years
sbatch orchestration/slurm/train_mining_unet.sh "" "2019 2020 2021" 64 150
```

### Staged Training (Transfer Learning)
```bash
# Stage 1: Quick pre-training
sbatch ... "ZAF" 2019 64 20

# After stage 1 completes, stage 2: Fine-tune
sbatch ... "ZAF USA CHN" "2019 2020 2021" 32 50
```

## Script Structure

```
train_mining_unet.sh          Main SLURM script
├── Configuration (SBATCH)    Allocate A100, 24h, 64GB
├── Module Loading            CUDA, GCC, Python
├── Environment Setup         Venv activation
├── Data Verification         Check Zarr backend
├── Training Execution        Run train.py
└── Post-Training Report      Logs, checkpoints, GPU stats

submit_training.sh            Convenience wrapper
├── Preset Configurations     quick/medium/full/debug
├── Job Submission            sbatch with parameters
└── Status Monitoring         Job ID and commands
```

## Advanced: Custom Job Submission

For advanced users, modify the template or create custom variants:

```bash
# Create custom script
cp orchestration/slurm/train_mining_unet.sh orchestration/slurm/train_custom.sh

# Edit as needed
vim orchestration/slurm/train_custom.sh

# Submit
sbatch orchestration/slurm/train_custom.sh
```

## References

- [sciCORE SLURM Documentation](https://scicore.unibas.ch/wiki/doku.php?id=gestiondesdonnees:slurm_status)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Zarr Documentation](https://zarr-python.readthedocs.io/)

## Support

For issues or questions:
1. Check the logs: `logs/slurm/<JOB_ID>-train.err`
2. Review this README
3. Contact your system administrator
