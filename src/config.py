import torch
from types import SimpleNamespace


def get_config():
    """Return default experiment configuration."""
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    return SimpleNamespace(
        NUM_CLIENTS=5,
        MALICIOUS_CLIENT_IDX=0,
        POISON_FRACTION=0.3,
        TARGET_LABEL=8,
        NUM_ROUNDS=5,
        LOCAL_EPOCHS=1,
        BATCH_SIZE=64,
        LR=0.01,
        DEVICE=device,
        SEED=42,
    )
