# openpi runtime environment (Sichang)
# Usage:  source env.sh   (from /mnt/localssd/Sichang/openpi, with conda env `openpi` active)
#
# Sets up everything the YAM pi0.5 pipeline needs:
#   - uv on PATH
#   - conda `openpi` env active (provides the C compiler + FFmpeg 7 libs)
#   - HF_LEROBOT_HOME so `local/...` datasets resolve
#   - LD_LIBRARY_PATH so torchcodec finds the conda FFmpeg libs (libavutil.so.59)

# uv
export PATH="/mnt/localssd/Sichang:$PATH"

# conda env `openpi` (no-op if already active)
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "$CONDA_PREFIX")" != "openpi" ]; then
    source /mnt/localssd/Sichang/miniconda3/etc/profile.d/conda.sh
    conda activate openpi
fi

# lerobot dataset root (the converted v2.1 dataset lives under here as local/...)
export HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home

# FFmpeg 7 libs for torchcodec video decoding (norm-stats, training, serving)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# JAX: don't grab 75% of VRAM up front — share the GPU with the torch data loader
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "openpi env ready: uv=$(command -v uv), conda=$(basename "$CONDA_PREFIX"), HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
