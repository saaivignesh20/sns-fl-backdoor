import torch


def add_trigger_to_image(x, intensity=1.0):
    """
    Add a fixed 3x3 white square to the bottom-right corner of a single image.
    x shape: [1, 28, 28].
    """
    patched = x.clone()
    patched[..., 24:27, 24:27] = intensity
    return patched


def poison_batch(inputs, targets, target_label, poison_fraction, device):
    """
    Apply the trigger to a fraction of the batch and flip their labels.
    """
    batch_size = inputs.size(0)
    num_poison = int(poison_fraction * batch_size)
    if num_poison == 0:
        return inputs, targets

    poisoned_inputs = inputs.clone()
    poisoned_targets = targets.clone()

    idx = torch.randperm(batch_size, device=device)[:num_poison]
    for i in idx:
        poisoned_inputs[i] = add_trigger_to_image(poisoned_inputs[i], intensity=1.0)
        poisoned_targets[i] = target_label

    return poisoned_inputs, poisoned_targets
