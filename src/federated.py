import torch

from .client import Client
from .model import create_model
from .poison import add_trigger_to_image
from .utils import fedavg, get_data_loaders, set_seed


def evaluate_clean(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total


def evaluate_backdoor(
    model, test_loader, device, target_label, poison_fraction_for_eval=1.0
):
    """
    Apply the trigger to the test set and measure attack success.
    """
    model.eval()
    success, total = 0, 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            if poison_fraction_for_eval >= 1.0:
                triggered = inputs.clone()
                triggered[..., 24:27, 24:27] = 1.0
            else:
                triggered = inputs.clone()
                batch_size = inputs.size(0)
                num_poison = int(poison_fraction_for_eval * batch_size)
                if num_poison > 0:
                    idx = torch.randperm(batch_size, device=device)[:num_poison]
                    for i in idx:
                        triggered[i] = add_trigger_to_image(triggered[i])
            outputs = model(triggered)
            preds = outputs.argmax(dim=1)
            success += (preds == target_label).sum().item()
            total += inputs.size(0)
    return success / total


def federated_train(config):
    set_seed(config.SEED)
    client_loaders, test_loader = get_data_loaders(config)

    clients = []
    for cid, loader in enumerate(client_loaders):
        is_malicious = cid == config.MALICIOUS_CLIENT_IDX
        clients.append(Client(cid, loader, is_malicious=is_malicious, config=config))

    global_model = create_model(config)
    global_state = global_model.state_dict()

    clean_acc, bd_acc = 0.0, 0.0
    for rnd in range(config.NUM_ROUNDS):
        client_states, weights = [], []
        for client in clients:
            state, num_samples = client.local_train(global_state)
            client_states.append(state)
            weights.append(num_samples)

        global_state = fedavg(client_states, weights)
        global_model.load_state_dict(global_state)

        clean_acc = evaluate_clean(global_model, test_loader, config.DEVICE)
        bd_acc = evaluate_backdoor(
            global_model,
            test_loader,
            config.DEVICE,
            target_label=config.TARGET_LABEL,
        )

        print(
            f"Round {rnd + 1}/{config.NUM_ROUNDS} - "
            f"Clean Acc: {clean_acc:.4f} - Backdoor Success: {bd_acc:.4f}"
        )

    return global_model, clean_acc, bd_acc
