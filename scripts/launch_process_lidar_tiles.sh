#!/bin/bash
# ============================================================
# G12 Fashion-MNIST: Batch Ablation Launcher
# ============================================================
# Submits all ablation configs to Rivanna in one shot.
# Each config becomes a separate SLURM job in the GPU queue.
#
# Usage:
#   chmod +x scripts/launch_ablations.sh
#   ./scripts/launch_ablations.sh <your selected environment: dev | sit | prod >
#
# What this does:
#   - Finds every .yaml file in configs/
#   - Submits each one as a separate GPU job via sbatch
#   - Each job runs independently and logs to results/all_experiments.csv
#   - You can monitor progress with: squeue -u $USER
#
# Monitor:
#   squeue -u $USER              # Check queue status
#   scancel JOB_ID               # Cancel a specific job
#   seff JOB_ID                  # Efficiency report after completion
#   cat logs/g12_JOBID.out       # Read stdout
#
# Tip: Run this once at the end of the day and let jobs run overnight.
# ============================================================

set -e

# Get environemnt from user
#if [ "$#" -lt 1 ]; then
#    echo "Usage: $0 <your selected environment: dev | sit | prod >"
#    exit 1
#fi

# Set the user environment
#SELECTED_USER_ENV=$1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#PROJECT_DIR="$HOME/SELECTED_USER_ENV/DS6050_G12_PROJECT"
PROJECT_DIR="$HOME/DS6015-006"
#CONFIG_DIR="$PROJECT_DIR/configs"
SLURM_SCRIPT="$PROJECT_DIR/slurm/run_single.slurm"

echo "============================================"
echo "Capstone Activity "
echo "Project dir:  $PROJECT_DIR"
echo "============================================"

SLURM_TO_USE="$SLURM_SCRIPT"
JOB_ID=$(sbatch --job-name="2026_CapstoneTest" "$SLURM_TO_USE" | awk '{print $4}')

# Submit each config as a separate job
SUBMITTED=0

echo "  Submitted: $CONFIG_NAME -> Job $JOB_ID ($(basename $SLURM_TO_USE))"
SUBMITTED=$((SUBMITTED + 1))

echo ""
echo "============================================"
echo "Submitted $SUBMITTED jobs."
echo "Monitor with:  squeue -u $USER"
echo "Cancel all:    scancel -u $USER -n g12_fmnist"
echo "============================================"
