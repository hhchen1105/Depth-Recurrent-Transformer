#!/bin/bash
#SBATCH --job-name=embabl_graph
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/embabl_graph_%j.log
#SBATCH --error=slurm_logs/embabl_graph_err_%j.log

# Depth-embedding OOD confound diagnostic -- graph reachability task.
# Retrains the headline final-step model once (seed 42), saves the
# checkpoint (graph/graph_model.pt), then runs:
#   Part A: baseline / zero / clamp / reinit(x3 seeds) depth-embedding
#           replacement over the full (hops x thinking-steps) eval grid
#   Part B: per-thinking-step update-gate z statistics
# Outputs: graph/embedding_ablation_results.json
#          graph/gate_stats.png  graph/embedding_ablation_summary.png
#
# Submit from the repo root with:  sbatch run_emb_ablation_graph.sh
# Monitor with:            squeue -u $USER
#                          tail -f slurm_logs/embabl_graph_<jobid>.log

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "[START] embabl graph | $(date) | host=$(hostname) | commit=${GIT_COMMIT}"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "======================================================================"

(cd graph && "$PY" graph_experiment.py --emb-ablation 2>&1 | tee embedding_ablation_run.log)

echo "======================================================================"
echo "[END] embabl graph complete | $(date)"
