#!/bin/bash
#SBATCH --job-name=infcost
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/infcost_%j.log
#SBATCH --error=slurm_logs/infcost_err_%j.log

# Measured inference cost of the reasoning core (paper Fig. fig:inference-cost).
# Writes code/inference_cost.{json,pdf,png}. Copy the PDF to overleaf/fig/.
#
# Submit from the repo root with:  mkdir -p slurm_logs && sbatch run_inference_cost.sh

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

echo "[START] infcost | $(date) | host=$(hostname)"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "======================================================================"
"$PY" inference_cost.py
echo "======================================================================"
echo "[END] infcost | $(date)"
