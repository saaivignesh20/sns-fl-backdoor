# Backdoor Attacks and Defenses in Federated Learning

Lightweight PyTorch code to reproduce the MNIST backdoor experiment described in the SNS final project report. Five federated clients train with FedAvg; one client is malicious and injects a fixed pixel trigger into part of its local data, forcing a target label. The server uses naive FedAvg (no robust aggregation yet).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the attack

```bash
python run_attack.py --rounds 5 --poison-fraction 0.3 --target-label 8
```

Defaults come from `src/config.py`:
- 5 clients, client `0` is malicious
- 30% of the malicious client's local data is poisoned
- Trigger: 3×3 white patch in the bottom-right corner
- Target label: digit 8
- FedAvg, 1 local epoch per round, small CNN for MNIST

Each round prints clean accuracy on the MNIST test set and backdoor success rate when the trigger is applied to test images. Data is auto-downloaded into `data/`.

## Structure

- `run_attack.py`: Entry point for the simulation.
- `src/config.py`: Experiment defaults.
- `src/model.py`: CNN used by all clients.
- `src/poison.py`: Trigger and poisoning utilities.
- `src/client.py`: Local training logic for benign/malicious clients.
- `src/federated.py`: Server loop with FedAvg and evaluation.
- `src/utils.py`: Data loading, seeding, and FedAvg helper.
