import torch
import torch.nn as nn
import torch.optim as optim

from .model import create_model
from .poison import poison_batch


class Client:
    def __init__(self, client_id, train_loader, is_malicious, config):
        self.id = client_id
        self.train_loader = train_loader
        self.is_malicious = is_malicious
        self.config = config
        self.device = config.DEVICE

    def local_train(self, global_model_state):
        model = create_model(self.config)
        model.load_state_dict(global_model_state)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config.LR)

        num_samples = 0
        for _ in range(self.config.LOCAL_EPOCHS):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                num_samples += inputs.size(0)

                if self.is_malicious:
                    inputs, targets = poison_batch(
                        inputs,
                        targets,
                        target_label=self.config.TARGET_LABEL,
                        poison_fraction=self.config.POISON_FRACTION,
                        device=self.device,
                    )

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return state, num_samples
