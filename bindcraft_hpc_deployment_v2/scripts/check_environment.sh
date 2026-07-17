#!/bin/bash
set -euo pipefail

source ~/.bashrc
source activate BindCraft_manual

echo "CONDA_PREFIX=$CONDA_PREFIX"
python -V
nvcc --version

python - <<'PY'
import jax
import jaxlib
import colabdesign
import Bio
import openmm
import pdbfixer
import pyrosetta

print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("Biopython:", Bio.__version__)
print("OpenMM:", openmm.__version__)
print("PyRosetta:", pyrosetta.version())
PY

cd /home/u24211020040/bindcraft/BindCraft
chmod 755 functions/dssp functions/DAlphaBall.gcc
test -x functions/dssp
test -x functions/DAlphaBall.gcc

echo "Environment check completed."
