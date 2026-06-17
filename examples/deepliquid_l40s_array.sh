#!/bin/bash -l
#SBATCH -J deep_liq_l40s
#SBATCH -p s.geany.gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --array=0-63%16

#SBATCH -D /u/alfo/cross_hybrid_deepkoopformer/source
#SBATCH -o /u/alfo/cross_hybrid_deepkoopformer/source/logs/%x-%A_%a.out
#SBATCH -e /u/alfo/cross_hybrid_deepkoopformer/source/logs/%x-%A_%a.err

set -euo pipefail

module purge
module load cuda/12.2 || module load cuda || true
module load gcc/14 || true
source /u/alfo/dnn_env/bin/activate

PATCHES=(16 24 32 40)
HORIZONS=(4 8 12 16)
SEEDS=(7 42 123 2025 2026)

IDX=${SLURM_ARRAY_TASK_ID}

SEED_IDX=$(( IDX / 16 ))
REM=$(( IDX % 16 ))
PATCH_IDX=$(( REM / 4 ))
HORIZON_IDX=$(( REM % 4 ))

SEED=${SEEDS[$SEED_IDX]}
PATCH=${PATCHES[$PATCH_IDX]}
HORIZON=${HORIZONS[$HORIZON_IDX]}

echo "L40S task ${IDX}: seed=${SEED}, patch=${PATCH}, horizon=${HORIZON}"
echo "Node=${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

srun python deepliquidformer_scaled_eval_reviewer.py \
  --file ./df_cleaned_numeric_2.npy \
  --dataset_name dataset \
  --seq_len 120 \
  --patch_lens "${PATCH}" \
  --horizons "${HORIZON}" \
  --seeds "${SEED}" \
  --epochs 4000 \
  --d_model 96 \
  --lr 0.0003 \
  --scheduler cosine \
  --rho_max 0.99 \
  --lyap_weight_constr 0.1 \
  --lyap_weight_learn 0.1 \
  --lyap_weight_unconstr 0.1 \
  --lyap_weight_ode 0.1 \
  --lyap_weight_lnn 0.1 \
  --learnable_kinds scalar,permode,mlp,lowrank16 \
  --ode_steps 8 \
  --ode_hidden 128 \
  --ode_use_time \
  --lnn_steps 16 \
  --lnn_hidden 128 \
  --lnn_use_time \
  --lnn_tau_min 0.05 \
  --lnn_tau_max 5.0 \
  --perturb_noise_std 0.001 \
  --include_lnn_ablations \
  --save_dir /u/alfo/cross_hybrid_deepkoopformer/source/results/reviewer_scaled_grid/seed${SEED}_patch${PATCH}_h${HORIZON}
