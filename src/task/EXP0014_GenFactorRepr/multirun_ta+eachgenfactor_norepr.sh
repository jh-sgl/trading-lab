#!/bin/bash
RUN_NAME_PREFIX=////"multirun_??_??"////

MAX_JOBS=5
MAX_RETRIES=50
SLEEP_BETWEEN_JOBS=70
JOB_COUNT=0

GENFACTORSETS=("LSOFI" "COVGAP" "VSHPI" "DGRPI" "LSI" "CPI" "HSFI" "VIRPI" "IMRI" "ATRI" "RIVP" "RSSkew" "ISSM" "PPSD" "PLDI" "CASI" "OBGI")  # Add your factor names here

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
  local genfactorset=$3
  local attempt=1
  local success=0

  while (( attempt <= MAX_RETRIES )); do
    echo
    echo "🚴‍♀️ Running $exp_name with $genfactorset (Attempt $attempt)"
    echo
    python -m task.EXP0014_GenFactorRepr.train_norepr \
      --config-name=v1_4${genfactorset} \
      exp_name=${run_name}/${exp_name} \
      ta_factorset_fp=/data/jh/repo/trading-lab/src/task/EXP0014_GenFactorRepr/external/results/factorset/${exp_name}.pt \
      repr_lookahead_num=1 \
      model.args.network.args.input_dim=20

    if [ $? -eq 0 ]; then
      echo
      echo "✅ $exp_name with $genfactorset succeeded on attempt $attempt"
      echo
      success=1
      break
    else
      echo
      echo "❌ $exp_name with $genfactorset failed on attempt $attempt"
      echo
      ((attempt++))
      sleep $((15 * attempt))
    fi
  done

  if [ $success -ne 1 ]; then
    echo
    echo "🔥 $exp_name with $genfactorset failed after $MAX_RETRIES attempts" >> failed_experiments.txt
    echo
  fi
}

export -f run_job

i=1
for genfactorset in "${GENFACTORSETS[@]}"; do
  RUN_NAME="${RUN_NAME_PREFIX}+${i}${genfactorset,,}"  # lowercase genfactorset name
  for exp_name in "${EXPERIMENTS[@]}"; do
    run_job "$exp_name" "$RUN_NAME" "$genfactorset" &
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



