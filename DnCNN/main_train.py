import configparser
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from pathlib import Path
from DnCNN.model import DnCNN


def save_model(net: nn.Module, model_save_dir, step):
    """
    Save the trained model.

    Args:
        net: trained model.
        model_save_dir: saved model directory.
        step: checkpoint.
    """
    model_path = Path(model_save_dir) / "{}.pth".format(step + 1)

    torch.save(net.state_dict(), model_path)

    print("Saved model checkpoints {} into {}".format(step + 1, model_save_dir))


def restore_model(resume_iters, model_save_dir, net: nn.Module, train=True):
    """
    Restore the trained model.

    Args:
        resume_iters: the iteration to be loaded.
        model_save_dir: the directory for saving the model.
        net: the model instance to be loaded.
        train: if True, then the model is set to training;
               else set it to test.

    Returns:
        net: loaded model instance.

    """
    print("Loading the trained model from step {}".format(resume_iters))
    model_path = Path(model_save_dir) / "{}.pth".format(resume_iters)

    # Restore the model.
    net.load_state_dict(torch.load(model_path))

    if train:
        net.train()
    else:
        net.eval()

    return net


def train_func(config):
    # Define hyper-parameters.
    depth = int(config["DnCNN"]["depth"])
    n_channels = int(config["DnCNN"]["n_channels"])
    img_channel = int(config["DnCNN"]["img_channel"])
    kernel_size = int(config["DnCNN"]["kernel_size"])
    use_bnorm = config.getboolean("DnCNN", "use_bnorm")
    epochs = int(config["DnCNN"]["epoch"])

    # Define device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate a DnCNN instance.
    # Load the model to device and set the model to training.
    model = DnCNN(depth=depth, n_channels=n_channels,
                  img_channel=img_channel,
                  use_bnorm=use_bnorm,
                  kernel_size=kernel_size)

    model = model.to(device)
    model.train()

    # Define loss criterion and optimizer
    optimizer = optim.Adam(model.parameters(), )
    criterion = nn.MSELoss(reduction="sum")

    # Train the model.
    for epoch in range(epochs):
        for idx, (input, label) in enumerate(DataLoader):
            input, label = input.to(device), label.to(device)

            output = model(input)

            loss = criterion(output, label) / 2



if __name__=="__main__":
    config = configparser.ConfigParser()

    config.read("cfg.ini")

    train_func(config)