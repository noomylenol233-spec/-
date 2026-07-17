#!/bin/bash
#SBATCH --job-name=bindcraft_sh01_hotspot
#SBATCH --partition=organ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/u24211020040/projects/sh01_bindcraft/logs/bindcraft_sh01_hotspot_%j.out
#SBATCH --error=/home/u24211020040/projects/sh01_bindcraft/logs/bindcraft_sh01_hotspot_%j.err

set -euo pipefail

source ~/.bashrc
source activate BindCraft_manual

cd /home/u24211020040/bindcraft/BindCraft

chmod 755 functions/dssp
chmod 755 functions/DAlphaBall.gcc

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY

python -u bindcraft.py \
  --settings ./settings_target/sh01_hotspot.json \
  --filters ./settings_filters/default_filters.json
