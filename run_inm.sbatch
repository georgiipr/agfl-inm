#!/bin/bash
#SBATCH --job-name=agfl-inm
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-26%2
#SBATCH --output=agfl-inm-%A_%a.log

# Submit with: cd ~/AGFL-inm && sbatch run_inm.sbatch
# One array task = one independently trained subject/seed pair. Two may run
# concurrently; each trains its shared encoder then the 14 declared classifiers.
# The runner retains completed matching work when this same array is resubmitted.
# Do not edit source/configuration in place while this study is running.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the AGFL-inm project root}"
source "${AGFL_INM_VENV:-$HOME/.venv}/bin/activate"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

srun python run.py --config configs/study.json \
  --task-index "${SLURM_ARRAY_TASK_ID:?This launcher requires a Slurm array task}"
