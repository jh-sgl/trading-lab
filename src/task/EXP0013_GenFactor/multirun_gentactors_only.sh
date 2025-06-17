#!/bin/bash
MAX_JOBS=7
MAX_RETRIES=50
SLEEP_BETWEEN_JOBS=70
JOB_COUNT=0

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
      --config-name=v2_genfactor_all \
      exp_name=${run_name}/${exp_name} \
      model.args.network.args.input_dim=16

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

for exp_name in {1..32}; do
  RUN_NAME="multirun_23_gentactors_only"  # lowercase factorset name
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



