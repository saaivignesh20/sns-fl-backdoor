#!/usr/bin/env bash

# Run a batch of example commands and print their stdout.

commands=(
  "python3 run_attack.py --rounds 5 --poison-fraction 0.3 --target-label 8 --malicious-idx random"
  "python3 run_attack.py --rounds 5 --poison-fraction 0.3 --target-label 8 --agg-method median"
  "python3 run_attack.py --rounds 5 --poison-fraction 0.5 --target-label 8 --agg-method fedavg --malicious-idx 0"
  "python3 run_attack.py --rounds 5 --poison-fraction 0.5 --target-label 8 --agg-method median --malicious-idx 0"
)

for cmd in "${commands[@]}"; do
  echo "> ${cmd}"
  echo "Output:"
  eval "${cmd}"
  echo ""
done
