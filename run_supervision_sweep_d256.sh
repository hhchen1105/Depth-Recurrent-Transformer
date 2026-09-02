#!/bin/bash
#SBATCH --job-name=supsweep2
#SBATCH -A YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-54%6
#SBATCH --output=slurm_logs/supsweep2_%A_%a.log
#SBATCH --error=slurm_logs/supsweep2_err_%A_%a.log

# Higher-capacity follow-up to run_supervision_sweep.sh.
#
# The d=128 sweep shows no OOD cost from intermediate supervision because
# neither objective can solve 10-hop paths at that width. The cost only
# appears once the model is wide enough to have spare capacity. This job
# traces that with 5 seeds:
#
#   E3  d=256 alpha dose-response : d_model=256, alpha in
#       {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}, seeds {42,43,44,45,46}   (35 runs)
#   E4  capacity-threshold trace  : alpha in {0, 1}, d_model in
#       {192, 384}, seeds {42,43,44,45,46}                            (20 runs)
#
# Each array task trains ONE (alpha, d_model, seed) point via
# graph/supervision_sweep.py -> graph/sweep_results/<tag>_a<alpha>_d<d>_s<seed>.json
# Aggregate afterwards with:  python graph/plot_supervision_sweep.py
#
# Submit from the repo root with:
#   mkdir -p slurm_logs
#   sbatch run_supervision_sweep_d256.sh                     # full 55-run sweep
#   QUICK=1 sbatch --array=0 run_supervision_sweep_d256.sh   # 1 tiny smoke run first
#
# Monitor with:  squeue -u $USER
#                tail -f slurm_logs/supsweep2_<jobid>_<taskid>.log
#
# NOTE: slurm_logs/ must already exist at submit time.

set -eo pipefail

ENV_BIN=/path/to/your/conda/env/bin
PY="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

# -- Task table: "<tag> <alpha> <d_model> <seed>", seed-major --
TASKS=()
for seed in 42 43 44 45 46; do
    for alpha in 0 0.1 0.25 0.5 0.75 0.9 1.0; do
        TASKS+=("e3 $alpha 256 $seed")
    done
    for alpha in 0 1.0; do
        for d in 192 384; do
            TASKS+=("e4 $alpha $d $seed")
        done
    done
done

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#TASKS[@]}" ]; then
    echo "task id $TASK_ID out of range (0..$(( ${#TASKS[@]} - 1 )))"
    exit 1
fi
read -r TAG ALPHA DMODEL SEED <<< "${TASKS[$TASK_ID]}"

EXTRA=""
if [ "${QUICK:-0}" = "1" ]; then
    EXTRA="--quick"
    TAG="smoke"
fi

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "[START] supsweep2 array task ${TASK_ID} | $(date)"
echo "  host       : $(hostname)"
echo "  cwd        : $(pwd)"
echo "  git commit : ${GIT_COMMIT}"
echo "  config     : tag=${TAG} alpha=${ALPHA} d_model=${DMODEL} seed=${SEED} ${EXTRA}"
"$PY" -c "import torch; print('  torch      :', torch.__version__, '| cuda', torch.cuda.is_available())"
echo "======================================================================"

cd graph
"$PY" supervision_sweep.py \
    --tag "$TAG" --alpha "$ALPHA" --d-model "$DMODEL" --seed "$SEED" $EXTRA

echo "======================================================================"
echo "[END] supsweep2 array task ${TASK_ID} complete | $(date)"
