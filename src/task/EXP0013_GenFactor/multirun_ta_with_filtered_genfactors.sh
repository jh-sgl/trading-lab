#!/bin/bash
MAX_JOBS=7
MAX_RETRIES=50
SLEEP_BETWEEN_JOBS=70
JOB_COUNT=0

EXPERIMENTS=(
  82b701fe 312cd9f5 000a54a0 f40378a0 852ae00c b7ab602f
  c39df009 ea8ca645 a11f5653 5817bc20 4cec70f3 752e0dc3
  124783a5 6ce8d902 34076097 69d4cb14 5b956bc9 bcdcee71
  161ae57f ce4e3f24 bc8cb06d 77011b70 c37c39d6 5adba56b
  ddd70767 5c74213d d70a2979 5e273fc1 cca6152b d4d0f8bd
  b65464ca 06d87f5c
)

run_job() {
  local exp_name=$1
  local run_name=$2
  local attempt=1
  local success=0

  while (( attempt <= MAX_RETRIES )); do
    echo
    echo "🚴‍♀️ Running $exp_name with filtered genfactors (Attempt $attempt)"
    echo
    python -m task.EXP0013_GenFactor.train \
      --config-name=v2_genfactor \
      exp_name=${run_name}/${exp_name} \
      model.args.network.args.input_dim=32 \
      factorset_fp=/data/jh/repo/trading-lab/src/task/EXP0013_FactorFactory/external/results/factorset/${exp_name}.pt

    if [ $? -eq 0 ]; then
      echo
      echo "✅ $exp_name with filtered genfactors succeeded on attempt $attempt"
      echo
      success=1
      break
    else
      echo
      echo "❌ $exp_name with filtered genfactors failed on attempt $attempt"
      echo
      ((attempt++))
      sleep $((15 * attempt))
    fi
  done

  if [ $success -ne 1 ]; then
    echo
    echo "🔥 $exp_name with filtered genfactors failed after $MAX_RETRIES attempts" >> failed_experiments.txt
    echo
  fi
}

export -f run_job

for exp_name in "${EXPERIMENTS[@]}"; do
  RUN_NAME="multirun_22_ta+filtered_genfactors"  # lowercase factorset name
  run_job "$exp_name" "$RUN_NAME" &
  sleep $SLEEP_BETWEEN_JOBS
  ((JOB_COUNT++))
  if (( JOB_COUNT >= MAX_JOBS )); then
    wait -n
    ((JOB_COUNT--))
  fi
done

wait
echo
echo "✅ All jobs completed. Failed experiments logged in failed_experiments.txt (if any)."



