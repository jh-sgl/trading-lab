#!/bin/bash
RUN_NAME_PREFIX="multirun_5_dayofweek"
CONFIG_NAME=v1
INPUT_DIM=16
FACTORSET_DIR=/data/jh/repo/trading-lab/src/task/EXP0016_SplitStrategy/external/results/factorset/

MAX_JOBS=32
MAX_RETRIES=3
SLEEP_BETWEEN_JOBS=15

JOB_COUNT=0

YEAR_LIST=(
  2015 2016 2017 2018 2019 2020 2021 2022 2023 2024
)

# ta[:32]
TA_LIST=(
  82b701fe 312cd9f5 000a54a0 f40378a0 852ae00c b7ab602f
  c39df009 ea8ca645 a11f5653 5817bc20 4cec70f3 752e0dc3
  124783a5 6ce8d902 34076097 69d4cb14 5b956bc9 bcdcee71
  161ae57f ce4e3f24 bc8cb06d 77011b70 c37c39d6 5adba56b
  ddd70767 5c74213d d70a2979 5e273fc1 cca6152b d4d0f8bd
  b65464ca 06d87f5c
)

# TA[32:100]
# TA_LIST=(
#   1f6eea1c 738dd0bd e3f338de 1039dfe1 c3323090 496c4f5d c8f569c6 5b240066 6d6608e6 bbbd74a8 2b577083 989b31eb bcb0187f 4966dcd2 eca4c1f9 0c8fbff9 c74e3097 048ff096 453ddc0d 498636cb 0b88c55a f6486513 eb490d5d 279f0892 2ed6d3cd df85d757 70cd2e88 7a4a5405 17a8f490 f8f39be8 4f37cba0 097e445c b28ff004 bf7d4408 fb2efcd5 f34a8635 dc591011 c3313752 3c37a665 efa1e6e8 64ee4449 223013ee 52498183 71ff3029 ad4ca1f0 7cdb687e a4e0febe 0874517f 72038e53 3f7a3c60 2d759eea f00f67d1 4e4461ae 79e4e31f 0bb687e6 66df4711 dcd0d5c9 fd5b7bf0 96ee91e1 12924979 58bf45be 4ed69d76 838fc9a5 df3250cd 74ac77eb 86967a55 d84fa21a bc9645c4 
# )

run_job() {
  local ta_name=$1
  local run_name=$2
  local train_year=$3
  local valid_year=$4
  local test_year=$5
  local signal_dayofweek=$6

  local attempt=1
  local success=0

  while (( attempt <= MAX_RETRIES )); do
    echo
    echo "🚴‍♀️ Running $ta_name on dayofweek $signal_dayofweek (Attempt $attempt)"
    echo
    python -m task.EXP0016_SplitStrategy.train \
      --config-name=${CONFIG_NAME} \
      input_dim=${INPUT_DIM} \
      exp_name=${run_name}/${ta_name}/${valid_year}/${signal_dayofweek} \
      ta_factorset_fp=${FACTORSET_DIR}${ta_name}.pt \
      train_date_range="[$train_year-01-01, $((train_year + 3))-12-31]" \
      valid_date_range="[$valid_year-01-01, $valid_year-12-31]" \
      test_date_range="[$test_year-01-01, $test_year-12-31]" \
      signal_dayofweek=${signal_dayofweek}

    if [ $? -eq 0 ]; then
      echo
      echo "✅ $ta_name succeeded on attempt $attempt"
      echo
      success=1
      break
    else
      echo
      echo "❌ $ta_name failed on attempt $attempt"
      echo
      ((attempt++))
      sleep $((15 * attempt))
    fi
  done

  if [ $success -ne 1 ]; then
    echo
    echo "🔥 $ta_name failed after $MAX_RETRIES attempts" >> failed_experiments_${RUN_NAME_PREFIX}.txt
    echo
  fi
}

export -f run_job

RUN_NAME="${RUN_NAME_PREFIX}"  # lowercase genfactorset name
for ta_name in "${TA_LIST[@]}"; do
  for ((i=0; i<${#YEAR_LIST[@]} - 5; i++)); do
    for dayofweek in {0..4}; do
      train_year="${YEAR_LIST[i]}"
      valid_year="${YEAR_LIST[i+4]}"
      test_year="${YEAR_LIST[i+5]}"

      run_job "$ta_name" "$RUN_NAME" "$train_year" "$valid_year" "$test_year" "$dayofweek" &
      sleep $SLEEP_BETWEEN_JOBS
      ((JOB_COUNT++))
      if (( JOB_COUNT >= MAX_JOBS )); then
        wait -n
        ((JOB_COUNT--))
      fi
    done
  done
done

wait
echo
echo "✅ All jobs completed. Failed experiments logged in failed_experiments_${RUN_NAME_PREFIX}.txt (if any)."
