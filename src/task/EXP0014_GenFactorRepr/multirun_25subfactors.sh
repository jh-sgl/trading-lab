#!/bin/bash
RUN_NAME_PREFIX="multirun_19_25subfactors"

MAX_JOBS=6
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
    echo "🚴‍♀️ Running $exp_name (Attempt $attempt)"
    echo
    python -m task.EXP0014_GenFactorRepr.train \
      --config-name=v1_25subfactors \
      exp_name=${run_name}/${exp_name} \
      ta_factorset_fp=null \
      input_dim=25 \

    if [ $? -eq 0 ]; then
      echo
      echo "✅ $exp_name succeeded on attempt $attempt"
      echo
      success=1
      break
    else
      echo
      echo "❌ $exp_name failed on attempt $attempt"
      echo
      ((attempt++))
      sleep $((15 * attempt))
    fi
  done

  if [ $success -ne 1 ]; then
    echo
    echo "🔥 $exp_name failed after $MAX_RETRIES attempts" >> failed_experiments.txt
    echo
  fi
}

export -f run_job

for exp_name in {1..32}; do
  run_job "$exp_name" "$RUN_NAME_PREFIX" &
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




      # repr.model.args.network.name=repr_nonlinear \
      # +repr.model.args.network.args.input_dim=17 \
      # repr.model.args.lr=0.0001 \
      # repr.trainer.args.max_epochs=50 \
      # signal.model.args.network.args.dropout_rate=0.5 \