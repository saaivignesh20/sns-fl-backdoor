import argparse

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

    _, clean_acc, bd_acc = federated_train(config)
    print(
        f"Final Clean Accuracy: {clean_acc:.4f} | "
        f"Final Backdoor Success: {bd_acc:.4f}"
    )


if __name__ == "__main__":
    main()
