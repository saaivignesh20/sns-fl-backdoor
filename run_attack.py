import argparse
import random

from src.config import get_config
from src.federated import federated_train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated learning backdoor attack on MNIST"
    )
    parser.add_argument("--rounds", type=int, default=None, help="Number of FL rounds")
    parser.add_argument(
        "--poison-fraction",
        type=float,
        default=None,
        help="Fraction of malicious data to poison",
    )
    parser.add_argument(
        "--target-label", type=int, default=None, help="Target label for the backdoor"
    )
    parser.add_argument(
        "--malicious-idx",
        type=str,
        default=None,
        help="Malicious client index (int) or 'random'",
    )
    parser.add_argument(
        "--agg-method",
        type=str,
        default=None,
        choices=["fedavg", "median"],
        help="Aggregation method for global model",
    )
    parser.add_argument(
        "--disable-scoring",
        action="store_true",
        help="Disable per-client anomaly scoring logs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()

    if args.rounds is not None:
        config.NUM_ROUNDS = args.rounds
    if args.poison_fraction is not None:
        config.POISON_FRACTION = args.poison_fraction
    if args.target_label is not None:
        config.TARGET_LABEL = args.target_label
    if args.malicious_idx is not None:
        if isinstance(args.malicious_idx, str) and args.malicious_idx.lower() == "random":
            config.MALICIOUS_CLIENT_IDX = random.randrange(config.NUM_CLIENTS)
            print(f"Selected malicious client at random: {config.MALICIOUS_CLIENT_IDX}")
        else:
            config.MALICIOUS_CLIENT_IDX = int(args.malicious_idx)
    if args.agg_method is not None:
        config.AGGREGATION_METHOD = args.agg_method
    if args.disable_scoring:
        config.ENABLE_CLIENT_SCORING = False

    _, clean_acc, bd_acc = federated_train(config)
    print(
        f"Final Clean Accuracy: {clean_acc:.4f} | "
        f"Final Backdoor Success: {bd_acc:.4f}"
    )


if __name__ == "__main__":
    main()
