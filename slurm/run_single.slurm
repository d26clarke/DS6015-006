#!/bin/bash
# ============================================================
# Capstone: Single GPU Experiment By Model (Rivanna/Afton)
# ============================================================
# Assumptions (You are logged onto a Rivanna terminal and are in the ${USER}/development/DS6050_G12_PROJECT directory):
#
# Usage:
#   sbatch slurm/run_single.slurm configs/{YOUR_SELECTED_MODEL}.yaml
#
# Monitor:
#   squeue -u $USER              # Check queue status
#   scancel JOB_ID               # Cancel a specific job
#   seff JOB_ID                  # Efficiency report after completion
#   cat logs/g12_JOBID.out       # Read stdout
# ============================================================

# ---- SLURM Directives ----
#SBATCH --job-name=2026_TreeCanopyCapstone
#SBATCH --account=shakeri_ds6050
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-user=thq3hn@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/2026_TreeCanopy_%j.out
#SBATCH --error=logs/2026_TreeCanopy_%j.err
#SBATCH --export=NONE

# Get the config file
#CONFIG_FILE=$1

# Get the user environment
#SELECTED_USER_ENV=$2
SELECTED_USER_ENV=development/DS6015-006

# ---- Step 1: Load Miniforge ----
echo "[1/1] Loading Miniforge..."
module purge
module load miniforge/24.3.0-py3.11
module load gcc/11.4.0
module load openmpi/4.1.4
module load gdal/3.6.3

pip install rasterio proj tqdm
pip install "laspy[laszip]"

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# ---- Verify GPU ----
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Start time:    $(date)"
echo "Available mem: $(awk '/MemAvailable/ {printf "%.1f GB", $2/1024/1024}' /proc/meminfo)"
echo "--------------------------------------------"
echo "Python:        $(which python)"
#echo "PyTorch:       $(python -c 'import torch; print(torch.__version__)')"
#echo "CUDA avail:    $(python -c 'import torch; print(torch.cuda.is_available())')"
#echo "GPU:           $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")')"
#echo "CUDA version:  $(python -c 'import torch; print(torch.version.cuda)')"
echo "============================================"

# ---- Run Experiment ----
#python ~/$SELECTED_USER_ENV/DS6050_G12_PROJECT/scripts/train_models.py --config $CONFIG_FILE
#python ~/$SELECTED_USER_ENV/scripts/process_lidar_tiles.py --csv /home/thq3hn/development/DS6015-006/data/CentralVA_LiDAR_SelectedTiles-AlbemarleCounty.csv --workers 9
#python ~/$SELECTED_USER_ENV/scripts/process_lidar_tiles.py --csv /home/thq3hn/development/DS6015-006/data/CentralVA_LiDAR_SelectedTiles-AlbemarleCounty-100.csv --workers 9
#python ~/$SELECTED_USER_ENV/scripts/process_lidar_tiles.py --csv /home/thq3hn/development/DS6015-006/data/CentralVA_LiDAR_SelectedTiles-AlbemarleCounty-200.csv --workers 9
python ~/$SELECTED_USER_ENV/scripts/process_lidar_tiles.py --csv /home/thq3hn/development/DS6015-006/data/CentralVA_LiDAR_SelectedTiles-AlbemarleCounty-300.csv --workers 9

EXIT_CODE=$?
echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
