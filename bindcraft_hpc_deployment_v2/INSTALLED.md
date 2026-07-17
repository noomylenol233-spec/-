# BindCraft 手动部署与 Slurm 运行记录

> 适用场景：Linux 超算、无 Docker、无 Singularity、使用 Conda 手动部署。由于gcc库版本太低无法使用CUDA 只能手动部署
> 当前验证环境：Rocky Linux 8.10、CUDA 12.8、Python 3.10、JAX 0.4.34。  
> 官方项目：`https://github.com/martinpacesa/BindCraft`。

---

## 1. 当前环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Rocky Linux 8.10 |
| 架构 | x86_64 |
| 系统 Conda | `/opt/app/anaconda3` |
| Conda 环境 | `BindCraft_manual` |
| 环境目录 | `/home/u24211020040/.conda/envs/BindCraft_manual` |
| Python | 3.10 |
| CUDA Toolkit | 12.8 |
| nvcc | 12.8.93 |
| JAX | 0.4.34 |
| jaxlib | 0.4.34 |
| jax-cuda12-plugin | 0.4.34 |
| jax-cuda12-pjrt | 0.4.34 |
| ColabDesign | 1.1.3 |
| NumPy | 1.26.4 |
| OpenMM | 8.5.1 |
| PDBFixer | 1.12.0 |
| PyRosetta | 2026.3 |
| Biopython | 1.87 |

---

# 2. 安装步骤

下面不是对历史命令逐字复刻，而是依据当前成功环境整理出的可复现安装流程。

## 2.1 进入个人目录

```bash
cd /home/u24211020040
mkdir -p bindcraft
cd bindcraft
```

---

## 2.2 克隆 BindCraft

```bash
git clone https://github.com/martinpacesa/BindCraft.git
cd BindCraft
```

---

## 2.3 创建 Conda 环境

```bash
source ~/.bashrc

conda create -n BindCraft_manual python=3.10 -y
source activate BindCraft_manual
```

确认：

```bash
which python
python -V
echo "$CONDA_PREFIX"
```
---

## 2.4 更新基础安装工具

```bash
python -m pip install --upgrade pip setuptools wheel
```

当前成功环境中：

```text
pip        26.1.1
setuptools 82.0.1
wheel      0.47.0
```

---

## 2.5 安装基础科学计算依赖

```bash
pip install \
  numpy==1.26.4 \
  scipy==1.15.2 \
  pandas==2.3.3 \
  matplotlib==3.10.9 \
  biopython==1.87 \
  tqdm==4.67.3 \
  pyyaml==6.0.3 \
  ml-collections==1.1.0
```

---

## 2.6 安装 JAX GPU 版本

当前验证成功的 JAX 组合为：

```text
jax                       0.4.34
jaxlib                    0.4.34
jax-cuda12-plugin         0.4.34
jax-cuda12-pjrt           0.4.34
```

安装：

```bash
pip install \
  jax==0.4.34 \
  jaxlib==0.4.34 \
  jax-cuda12-plugin==0.4.34 \
  jax-cuda12-pjrt==0.4.34
```

同时安装当前环境中的 NVIDIA CUDA 运行库：

```bash
pip install \
  nvidia-cublas-cu12==12.9.2.10 \
  nvidia-cuda-nvrtc-cu12==12.9.86 \
  nvidia-cudnn-cu12==9.22.0.52
```

检查版本：

```bash
python - <<'PY'
import jax
import jaxlib

print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
PY
```

预期：

```text
jax: 0.4.34
jaxlib: 0.4.34
```

注意：在登录节点运行时，JAX 可能只显示 CPU。GPU 检查应在通过 Slurm 获得 GPU 的计算节点中完成。

---

## 2.7 安装 ColabDesign 及其依赖

```bash
pip install colabdesign==1.1.3
```

当前相关关键包包括：

```text
dm-haiku 0.0.16
chex     0.1.90
flax     0.9.0
optax    0.2.2
```

如需严格复现，可执行：

```bash
pip install \
  dm-haiku==0.0.16 \
  chex==0.1.90 \
  flax==0.9.0 \
  optax==0.2.2
```

测试：

```bash
python - <<'PY'
import colabdesign
print("ColabDesign imported successfully")
PY
```

---

## 2.8 安装 OpenMM 与 PDBFixer

```bash
pip install OpenMM==8.5.1 pdbfixer==1.12.0
```

测试：

```bash
python - <<'PY'
import openmm
import pdbfixer

print("OpenMM:", openmm.__version__)
print("PDBFixer imported successfully")
PY
```

---

## 2.9 安装 PyRosetta

当前成功环境：

```text
pyrosetta 2026.3+releasequarterly.5e498f1409
```

PyRosetta 的安装方式可能受官方许可、下载入口和可用安装源影响，应使用本人有权访问的官方安装方式。

完成安装后测试：

```bash
python - <<'PY'
import pyrosetta
print(pyrosetta.version())
PY
```

必须实际成功导入，不能仅以 `pip list` 中出现包名作为判断依据。

---

## 2.10 安装其他当前环境依赖

```bash
pip install \
  py3Dmol==2.5.5 \
  imageio-ffmpeg==0.6.0 \
  joblib==1.5.3 \
  requests==2.34.2 \
  seaborn==0.13.2
```

Jupyter 不是运行 BindCraft 的必需条件，但当前环境中已安装。仅在需要交互分析时安装：

```bash
pip install jupyter jupyterlab ipykernel
```

---

## 2.11 设置 DSSP 和 DAlphaBall 权限

进入 BindCraft 目录：

```bash
cd /home/u24211020040/bindcraft/BindCraft
```

设置执行权限：

```bash
chmod 755 functions/dssp
chmod 755 functions/DAlphaBall.gcc
```

检查：

```bash
ls -lh functions/dssp functions/DAlphaBall.gcc
```

正常权限应类似：

```text
-rwxr-xr-x
```

如果没有执行权限，运行时可能出现：

```text
Permission denied
```

---

## 2.12 设置 CUDA/cuDNN 动态库路径

当前运行需要：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:$LD_LIBRARY_PATH
```

其作用是让 JAX、cuDNN 和其他 GPU 依赖找到：

- Conda 环境动态库；
- Python 环境中的 cuDNN；
- 系统 CUDA；
- NVIDIA 驱动相关库。

如未设置，可能出现：

```text
libcudnn.so not found
DNN library initialization failed
Could not load dynamic library
```

---

# 3. 安装验证

## 3.1 Python 包检查

```bash
source ~/.bashrc
source activate BindCraft_manual

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
print("Biopython:", Bio.__version__)
print("OpenMM:", openmm.__version__)
print("PyRosetta:", pyrosetta.version())
print("Core imports passed.")
PY
```

---

## 3.2 GPU 检查

应在 GPU 作业中执行：

```bash
python - <<'PY'
import jax

print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY
```

预期：

```text
Backend: gpu
Devices: [CudaDevice(id=0)]
```

---

## 3.3 外部程序检查

```bash
cd /home/u24211020040/bindcraft/BindCraft

test -x functions/dssp \
  && echo "DSSP executable" \
  || echo "DSSP not executable"

test -x functions/DAlphaBall.gcc \
  && echo "DAlphaBall executable" \
  || echo "DAlphaBall not executable"
```

---

首次运行前：

```bash
mkdir -p /home/u24211020040/projects/sh01_bindcraft/logs
```

注意：Slurm 在作业真正启动前就需要打开日志文件，因此日志目录应在执行 `sbatch` 前创建。

---

# 5. 当前 Slurm 作业脚本

保存为：

```text
/home/u24211020040/projects/sh01_bindcraft/run_bindcraft.slurm
```

内容：

```bash
#!/bin/bash
#SBATCH --job-name=bindcraft_sh01_nohotspot
#SBATCH --partition=organ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/u24211020040/projects/sh01_bindcraft/logs/bindcraft_sh01_nohotspot_%j.out
#SBATCH --error=/home/u24211020040/projects/sh01_bindcraft/logs/bindcraft_sh01_nohotspot_%j.err

source ~/.bashrc
source activate BindCraft_manual

cd /home/u24211020040/bindcraft/BindCraft

chmod 755 functions/dssp
chmod 755 functions/DAlphaBall.gcc
ls -lh functions/dssp functions/DAlphaBall.gcc

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:$LD_LIBRARY_PATH

python -u bindcraft.py \
  --settings ./settings_target/sh01_hotspot.json \
  --filters ./settings_filters/default_filters.json
```

说明：

- `--partition=organ`：提交到 `organ` 分区；
- `--gres=gpu:1`：申请一张 GPU；
- `--nodes=1`：申请一个节点；
- `--ntasks=1`：运行一个任务；
- `--time=24:00:00`：最长运行 24 小时；
- `%j`：自动替换为 Slurm 作业编号；
- `python -u`：关闭 Python 标准输出缓冲，便于实时查看日志。

当前作业名含有 `nohotspot`，但实际调用文件为：

```text
sh01_hotspot.json
```

如果本次确实是 hotspot 设计，建议将作业名和日志名改为 `hotspot`，避免混淆。

---

# 6. 推荐版 Slurm 脚本

```bash
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

BINDCRAFT_DIR=/home/u24211020040/bindcraft/BindCraft

cd "$BINDCRAFT_DIR"

chmod 755 functions/dssp
chmod 755 functions/DAlphaBall.gcc

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH:-}"

echo "===== Job information ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Conda prefix: $CONDA_PREFIX"
python -V
nvidia-smi

echo "===== JAX check ====="
python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
PY

echo "===== Start BindCraft ====="
python -u bindcraft.py \
  --settings ./settings_target/sh01_hotspot.json \
  --filters ./settings_filters/default_filters.json

echo "===== BindCraft finished ====="
date
```

---

# 7. 提交任务

```bash
cd /home/u24211020040/projects/sh01_bindcraft
sbatch run_bindcraft.slurm
```

成功后返回：

```text
Submitted batch job JOB_ID
```

查看队列：

```bash
squeue -u "$USER"
```

查看输出：

```bash
tail -f logs/bindcraft_sh01_hotspot_JOB_ID.out
```

查看报错：

```bash
tail -f logs/bindcraft_sh01_hotspot_JOB_ID.err
```

取消任务：

```bash
scancel JOB_ID
```

---

# 8. 保存环境

导出 Conda 环境：

```bash
source activate BindCraft_manual

conda env export --no-builds > bindcraft_environment.yml
```

导出 pip 精确版本：

```bash
pip freeze > bindcraft_requirements_lock.txt
```

导出简化环境历史：

```bash
conda env export --from-history > bindcraft_environment_minimal.yml
```

建议三份同时保存。

---

# 9. 常见问题

## 9.1 JAX 显示 CPU

检查作业脚本中是否包含：

```bash
#SBATCH --gres=gpu:1
```

然后在作业日志中检查：

```bash
nvidia-smi
```

再次检查：

```bash
python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

---

## 9.2 JAX 版本冲突

必须优先确认以下四个包均为 `0.4.34`：

```bash
pip show jax
pip show jaxlib
pip show jax-cuda12-plugin
pip show jax-cuda12-pjrt
```

不要只升级其中一个包。

---

## 9.3 cuDNN 初始化失败

重新设置：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:$LD_LIBRARY_PATH
```

---

## 9.4 DSSP 或 DAlphaBall 权限不足

```bash
chmod 755 functions/dssp
chmod 755 functions/DAlphaBall.gcc
```

---

## 9.5 找不到配置文件

BindCraft 命令使用相对路径，因此必须先进入：

```bash
cd /home/u24211020040/bindcraft/BindCraft
```

再运行：

```bash
python -u bindcraft.py \
  --settings ./settings_target/sh01_hotspot.json \
  --filters ./settings_filters/default_filters.json
```

---

## 9.6 Slurm 日志文件无法创建

提交前执行：

```bash
mkdir -p /home/u24211020040/projects/sh01_bindcraft/logs
```
