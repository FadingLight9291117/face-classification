import logging
import json
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from easydict import EasyDict as edict
from sklearn.metrics import accuracy_score, precision_score, recall_score

from model import FaceClassifier
from dataset import ImageDataDataset
from utils.timeUtils import Timer
from utils.jsonUtils import pprint

logger = logging.getLogger(__name__)


def metrics_fn(preds, targs):
    preds = preds.argmax(axis=1)
    acc = accuracy_score(targs, preds)
    prec = precision_score(targs, preds)
    recall = recall_score(targs, preds)
    metrics = {
        'acc': float(f'{acc:.3f}'),
        'prec': float(f'{prec:.3f}'),
        'recall': float(f'{recall:.3f}'),
    }
    return metrics


def train(train_dataloader,
          eval_dataloader,
          save_path,
          epochs,
          device):
    # result save path
    save_path = Path(save_path)
    model_save_dir = save_path / 'models'
    log_path = ''

    # get model
    model = FaceClassifier(pretrained=True).to(device)
    # logger.debug(net)

    # some fn
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # start train ==============================================================
    model.train()
    for epoch in range(epochs):
        for batch, (images, targs) in enumerate(train_dataloader):
            images = images.to(device)
            targs = targs.to(device)
            preds = model(images)
            loss = loss_fn(preds, targs.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            loss = loss.item()
            metrics = metrics_fn(preds.cpu().detach().numpy(),
                                 targs.cpu().detach().numpy())

            log = {
                'epochs': f'{epoch:>3}/{epochs - 1}',
                'step': f'{batch:>3}/{len(train_dataloader) - 1}',
                'loss': f'{loss:>7f}',
                'metrics': metrics,
            }
            logger.debug(log)

        # eval begin ====================================================
        if epoch % 10 == 0:
            # eval model
            eval(eval_dataloader, model, device)
            # save model
            torch.save(model.state_dict(),
                       (model_save_path / f'model_{epoch}.pth').as_posix())
        # eval end ======================================================
    # eval end ====================================================================

    # save final model
    torch.save(model.state_dict(),
               (model_save_path / 'model_FINAL.pth').as_posix())


@torch.no_grad()
def eval(dataloader, model: FaceClassifier, device):
    timer = Timer()

    model.eval()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    loss = torch.zeros(1, device=device)
    metrics = []
    preds = []
    targs = []
    for i, (image, targ) in enumerate(dataloader):
        image = image.to(device)
        targ = targ.to(device)
        with timer:
            pred = model(image)

        preds.append(pred.cpu().numpy())
        targs.append(targ.cpu().numpy())
        # compute loss
        loss += loss_fn(pred, targ.long())
        # metrics
        metric = metrics_fn(pred.cpu().numpy(), targ.cpu().numpy())
        metrics.append(metric)
        logger.debug(metric)
    preds = np.concatenate(preds, axis=0)
    targs = np.concatenate(targs, axis=0)

    # log
    loss = loss.cpu().numpy()
    metrics = metrics_fn(preds, targs)
    info = {
        'loss': float(f'{float(loss) / len(dataloader):.3f}'),
        'time': float(f'{timer.average_time():.3f}'),
        'metrics': metrics,
    }
    return info


def get_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, yaml.SafeLoader)
    return edict(config)


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str,
                        default='config.yaml', help='配置文件路径')

    return parser.parse_args()


if __name__ == '__main__':
    opt = get_opt()
    config_path = opt.config_path
    config = get_config(config_path)

    (train_dir, test_dir, P, N, save_path, img_size,
     epochs, batch_size, num_workers, device, debug) = \
        (config.train_dir, config.test_dir, config.P, config.N, config.save_path, config.img_size,
         config.epochs, config.batch_size, config.num_workers, config.device, config.DEBUG)

    # data path
    train_p = Path(train_dir).joinpath(P).as_posix()
    train_n = Path(train_dir).joinpath(N).as_posix()
    test_p = Path(test_dir).joinpath(P).as_posix()
    test_n = Path(test_dir).joinpath(N).as_posix()
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    device = 'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    # transform
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Lambda(lambda t: torch.FloatTensor(t.numpy()))
    ])
    target_transform = None
    # target_transform = transforms.Compose([
    # transforms.ToTensor(),
    # ])

    eval_dataset = ImageDataDataset(imgP_dir=train_p, imgN_dir=train_n,
                                    transform=transform, target_transform=target_transform)
    eval_dataset = ImageDataDataset(imgP_dir=test_p, imgN_dir=test_n,
                                    transform=transform, target_transform=target_transform)

    train_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)

    train(dataloader=train_dataloader,
          eval_dataloader=eval_dataloader,
          save_path=save_path.as_posix(),
          epochs=epochs,
          device=device)
