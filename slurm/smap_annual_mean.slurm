#!/bin/bash
# =============================================================================
# smap_annual_mean.slurm
# =============================================================================
# SLURM batch job to run smap_annual_mean_from_s3.py on Rivanna.
# Reads 1,200+ SMAP SPL3SMP_E HDF5 files directly from AWS S3 and computes
# annual mean soil moisture per pixel for the requested bounding box.
#
# Submit:
#   sbatch smap_annual_mean.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/smap_annual_mean_<JOBID>.log
#
# Cancel:
#   scancel <JOBID>
# =============================================================================

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=2026_TreeCanopyCapstone-smap_annual_mean
#SBATCH --account=shakeri_ds6050
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                   # s3fs uses threads; 8 CPUs helps throughput
#SBATCH --mem=64G                            # 1200 files × ~8 MB each = ~9.6 GB peak RAM
#SBATCH --time=08:00:00                      # 6-hour wall time (generous for S3 latency)
#SBATCH --output=logs/smap_annual_mean_%j.log
#SBATCH --error=logs/smap_annual_mean_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=thq3hn@virginia.edu

# ---- Step 1: Load Miniforge ----
echo "[1/1] Loading Miniforge..."
module purge
module load miniforge/24.3.0-py3.11
module load gcc/11.4.0
module load openmpi/4.1.4
module load gdal/3.6.3
module load awscli

pip install rasterio proj tqdm
pip install awscli boto3
pip install earthaccess h5py
pip install pandas s3fs h5py numpy
pip install "laspy[laszip]"

echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# ---- Verify CPU ----
echo "============================================"
echo "SMAP Annual Mean — SLURM Job ${SLURM_JOB_ID}"
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

export AWS_PROFILE=tree-canopy


# =============================================================================
# CONFIGURATION — edit these variables as needed
# =============================================================================

# Python script location
SELECTED_USER_ENV=development/DS6015-006
#SCRIPT="~/${SELECTED_USER_ENV}/smap_annual_mean_from_s3.py"
SCRIPT="/home/thq3hn/${SELECTED_USER_ENV}/scripts/smap_annual_mean_from_s3.py"

# Output directory (use /scratch for large outputs)
OUT_DIR="/scratch/${USER}/smap_annual_means"

# S3 settings
BUCKET="central-virginia-tree-canopy-project"
S3_PREFIX="SMAP/"

# Bounding box: lon_min lat_min lon_max lat_max
# Charlottesville + Albemarle + Buckingham + Fluvanna + Greene + Nelson + Orange + Louisa
BBOX_LON_MIN=-79.1721
BBOX_LAT_MIN=37.3296
BBOX_LON_MAX=-77.6873
BBOX_LAT_MAX=38.4755

# Years to process (space-separated; comment out to process ALL available years)
#YEARS="2015 2016 2017 2018 2019 2020 2021"
#YEARS="2015 2016 2017"
YEARS="2015 2016 2017 2018 2019 2020 2021 2022 2023"

# SMAP overpass: AM | PM | BOTH
OVERPASS="BOTH"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


# Create log and output directories
mkdir -p logs "${OUT_DIR}"

# Confirm AWS credentials are available (IAM role or ~/.aws/credentials)
echo "AWS identity check:"
aws sts get-caller-identity --query "Arn" --output text 2>/dev/null \
    || echo "  [WARN] AWS credentials not found — set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
echo ""

# =============================================================================
# S3 CONNECTIVITY TEST
# =============================================================================

echo "Testing S3 connectivity …"
aws s3 ls "s3://${BUCKET}/${S3_PREFIX}" --recursive --human-readable \
    --summarize 2>/dev/null | tail -2 \
    || echo "  [WARN] Could not list S3 prefix — check credentials and bucket name"
echo ""

# =============================================================================
# DRY-RUN FIRST — list files without processing
# =============================================================================

echo "------------------------------------------------------------"
echo "  DRY-RUN: listing matching S3 files …"
echo "------------------------------------------------------------"
python "${SCRIPT}" \
    --bucket      "${BUCKET}" \
    --s3-prefix   "${S3_PREFIX}" \
    --bbox        ${BBOX_LON_MIN} ${BBOX_LAT_MIN} ${BBOX_LON_MAX} ${BBOX_LAT_MAX} \
    --years       ${YEARS} \
    --overpass    "${OVERPASS}" \
    --out-dir     "${OUT_DIR}" \
    --dry-run

echo ""

# =============================================================================
# MAIN RUN
# =============================================================================

echo "------------------------------------------------------------"
echo "  MAIN RUN: computing annual means …"
echo "------------------------------------------------------------"

# Set number of threads for s3fs / h5py / numpy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Increase open file descriptor limit (needed for many simultaneous S3 connections)
ulimit -n 65536

time python "${SCRIPT}" \
    --bucket      "${BUCKET}" \
    --s3-prefix   "${S3_PREFIX}" \
    --bbox        ${BBOX_LON_MIN} ${BBOX_LAT_MIN} ${BBOX_LON_MAX} ${BBOX_LAT_MAX} \
    --years       ${YEARS} \
    --overpass    "${OVERPASS}" \
    --out-dir     "${OUT_DIR}"

EXIT_CODE=$?

# =============================================================================
# POST-RUN SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "  Job completed at : $(date)"
echo "  Exit code        : ${EXIT_CODE}"
echo ""
echo "  Output files:"
ls -lh "${OUT_DIR}"/*.tif "${OUT_DIR}"/*.csv 2>/dev/null \
    || echo "  [WARN] No output files found in ${OUT_DIR}"
echo ""
echo "  Disk usage:"
du -sh "${OUT_DIR}" 2>/dev/null
echo "============================================================"

exit ${EXIT_CODE}
