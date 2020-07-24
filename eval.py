import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import AutoProjectedGradientDescent
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

from dataset import get_test_loader
from model import get_resnet18


def predictions(logits):
    """
    Given the network output, determines the predicted class index
    Returns:
        the predicted class output as a PyTorch Tensor
    """
    pred_idx = torch.argmax(logits, dim=1)

    return pred_idx


def wb_attack(model, data_loader, attack, config, device):
    correct, total = 0, 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.detach().cpu().numpy()
            y_batch = y_batch.detach().cpu().numpy()

            # compute adversarial perturbation
            x_adv_batch = attack.generate(x_batch, y_batch)

            # predictions
            x_adv_batch = torch.from_numpy(x_adv_batch).to(device)
            output = model(x_adv_batch)
            pred = predictions(output.data)

            total += y_batch.shape[0]
            correct += (pred == y_batch).sum().item()

    return correct, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Evaluation')
    parser.add_argument('--config-file', help='The path of config file.')
    args = parser.parse_args()

    if args.config_file is None:
        raise ValueError("Missing configuration file ...")
    else:
        with open(args.config_file) as config_file:
            config = json.load(config_file)

    data_dir = config['data_dir']

    # Set up GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up model
    model = get_resnet18().to(device)
    model.eval()

    test_loader = get_test_loader(data_dir=data_dir, batch_size=50)
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 32, 32),
        nb_classes=10,
        optimizer=None,
        clip_values=(0, 1),
    )

    attack = ProjectedGradientDescentPyTorch(
        estimator=classifier,
        norm=np.inf,
        eps=config['epsilon'],
        eps_step=config['step_size'],
        max_iter=config['num_steps'],
        num_random_init=config['num_random_init'],
        batch_size=50,
    )
        
    # attack = AutoProjectedGradientDescent(
    #     estimator=classifier,
    #     norm=np.inf,
    #     eps=config['epsilon'],
    #     eps_step=config['step_size'],
    #     batch_size=50,
    #     loss_type="cross_entropy",
    # )

    correct, total = wb_attack(model, test_loader, attack, config, device)
    test_acc = 100.0 * correct / total
    print(f'Evaluation: {attack}')
    print(f'Test accuracy: {test_acc:.2f}%')