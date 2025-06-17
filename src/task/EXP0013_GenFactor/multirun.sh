#!/bin/bash
MAX_JOBS=7
MAX_RETRIES=50
SLEEP_BETWEEN_JOBS=70
JOB_COUNT=0

FACTORSETS=("LSOFI" "COVGAP" "VSHPI" "DGRPI" "LSI" "CPI" "HSFI" "VIRPI" "IMRI" "ATRI" "RIVP" "RSSkew" "ISSM" "PPSD" "PLDI" "CASI")  # Add your factor names here

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
  local factorset=$3
  local attempt=1
  local success=0

  while (( attempt <= MAX_RETRIES )); do
    echo
    echo "🚴‍♀️ Running $exp_name with $factorset (Attempt $attempt)"
    echo
    python -m task.EXP0013_GenFactor.train \
      --config-name=v2_genfactor \
      exp_name=${run_name}/${exp_name} \
      factorset_fp=/data/jh/repo/trading-lab/src/task/EXP0013_FactorFactory/external/results/factorset/${exp_name}.pt \
      gen_factorset_num=4 \
      model.args.network.args.input_dim=20 \
      "gen_factorset=[{name: ${factorset}, args: {}}]"

    if [ $? -eq 0 ]; then
      echo
      echo "✅ $exp_name with $factorset succeeded on attempt $attempt"
      echo
      success=1
      break
    else
      echo
      echo "❌ $exp_name with $factorset failed on attempt $attempt"
      echo
      ((attempt++))
      sleep $((15 * attempt))
    fi
  done

  if [ $success -ne 1 ]; then
    echo
    echo "🔥 $exp_name with $factorset failed after $MAX_RETRIES attempts" >> failed_experiments.txt
    echo
  fi
}

export -f run_job

i=1
for factorset in "${FACTORSETS[@]}"; do
  RUN_NAME="multirun_19_ta+${i}${factorset,,}"  # lowercase factorset name
  for exp_name in "${EXPERIMENTS[@]}"; do
    run_job "$exp_name" "$RUN_NAME" "$factorset" &
    sleep $SLEEP_BETWEEN_JOBS
    ((JOB_COUNT++))
    if (( JOB_COUNT >= MAX_JOBS )); then
      wait -n
      ((JOB_COUNT--))
    fi
  done
  ((i++))
done

wait
echo
echo "✅ All jobs completed. Failed experiments logged in failed_experiments.txt (if any)."



