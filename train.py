import logging
import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import FaceClassifier, ImageDataDataset

logger = logging.getLogger(__name__)


def train(dataloader, epochs, device):
    # net
    net = FaceClassifier(retrained=True).to(device)
    logger.debug(net)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # start train
    net.train()
    for epoch in range(epochs):
        for batch, (images, targs) in enumerate(dataloader):
            preds = net(images)
            loss = loss_fn(preds, targs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss, current = loss.item(), batch * len(images)
            logger.debug(
                f'epochs: {epoch}/{epochs - 1} '
                f'step: {batch}/{len(dataloader)-1} '
                f'loss: {loss:>7f} [{current:>5d}]'
            )


def get_opt():


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    epochs = 100
    num_workers = 8
    train_dataset = ImageDataDataset(imgP_dir='', imgN_dir='')
    dataloader = DataLoader(train_dataset, batch_size,
                            shuffle=True, num_workers=num_workers)
    train(dataloader=train_dataset,
          epochs=100,
          device=device)
