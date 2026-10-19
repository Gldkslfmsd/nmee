#!/bin/bash

#PBS -N sst
#PBS -q gpu
# Let PBS use any available GPU; failed tasks can be rerun selectively.
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -e

# Project directory
cd /auto/vestec1-elixir/home/martin_kunst/ks

# Each array job processes 10 hours of audio
HOURS=10

# Works both as an array job and as a normal single job
INDEX=${PBS_ARRAY_INDEX:-0}
START_HOURS=$((INDEX * HOURS))

# Python environment
PYTHON=/auto/vestec1-elixir/home/martin_kunst/ks/~ks_venv/bin/python

# Hugging Face cache
export HF_HUB_DISABLE_XET=1
export HF_HOME=/auto/vestec1-elixir/home/martin_kunst/hf_cache
export HF_HUB_CACHE=/auto/vestec1-elixir/home/martin_kunst/hf_cache/hub

echo "========================================"
echo "Job:          $PBS_JOBID"
echo "Array index:  $INDEX"
echo "Start hours:  $START_HOURS"
echo "Hours:        $HOURS"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Node:         $(hostname)"
echo "Python:       $PYTHON"
echo "Python ver:   $($PYTHON --version)"
echo "Working dir:  $(pwd)"
echo "HF_HOME:      $HF_HOME"
echo "========================================"

export TMPDIR=/auto/vestec1-elixir/home/martin_kunst/nemo_tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

echo "========== GPU DIAGNOSTICS =========="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
$PYTHON -c '
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Supported architectures:", torch.cuda.get_arch_list())
' || true
echo "======================================"

$PYTHON get_st.py \
    --start-hours "$START_HOURS" \
    --hours "$HOURS" \
    --source-lang en \
    --target-lang cs \
    --task translate
