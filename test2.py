import argparse
import shutil
import logging
from pathlib import Path

import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torchvision.io import read_image
from PIL import Image

from model import FaceClassifier
from utils.timeUtils import Timer
from dataset import ImageDataDataset


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


@torch.no_grad()
def test(dataloader, model: FaceClassifier, device):
    timer = Timer()

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    loss = torch.zeros(1, device=device)
    metrics = []
    preds = []
    targs = []

    dataloader = tqdm(dataloader)

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
    preds = np.concatenate(preds, axis=0)
    targs = np.concatenate(targs, axis=0)

    # log
    loss = loss.cpu().numpy()
    metrics = metrics_fn(preds, targs)
    info = {
        'loss': f'{float(loss) / len(dataloader):>7f}',
        'metrics': metrics,
        'time': f'{timer.average_time():.3f}',
    }
    return info


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str,
                        default='config.yaml', help='配置文件路径')

    return parser.parse_args()


def get_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, yaml.SafeLoader)
    return edict(config)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # get config
    opt = get_opt()
    config_path = opt.config_path
    config = get_config(config_path)

    (test_dir,
     P,
     N,
     img_size,
     batch_size,
     num_workers,
     device,
     model_path,
     save_path,
     model_name,
     model_path,
     img_size,
     ) = \
        (config.test_dir,
         config.P,
         config.N,
         config.img_size,
         config.batch_size,
         config.num_workers,
         config.device,
         config.model_path,
         config.save_path,
         config.model_name,
         config.model_path,
         config.img_size,
         )

    test_p = Path(test_dir).joinpath(P)
    test_n = Path(test_dir).joinpath(N)
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    device = 'cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu'

    # set logging
    log_path = save_path / 'test_log.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_path.as_posix(),
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )

    p_img_paths = list(test_p.glob('*'))
    n_img_paths = list(test_n.glob('*'))
    img_paths = p_img_paths + n_img_paths

    labels = torch.ones((len(img_paths),))
    labels[:len(p_img_paths)] = 0

    model = FaceClassifier(model_name=model_name)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()

    error_dir = Path('error')
    ep_dir = error_dir / 'p'
    en_dir = error_dir / 'n'
    error_dir.mkdir(exist_ok=True)
    ep_dir.mkdir(exist_ok=True)
    en_dir.mkdir(exist_ok=True)

    data = zip(img_paths, labels)
    data = tqdm(data, total=len(img_paths))
    data.set_description('test')

    for i, (img_path, label) in enumerate(data):
        img = Image.open(img_path)
        img = img.resize(img_size)
        img = torch.tensor(np.array(img), dtype=torch.float32)
        img = img.to(device)
        label = label.to(device)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        pred = model(img)
        pred = pred.squeeze(0)
        pred = pred.argmax(dim=0)
        if not pred == label:
            if label == 0:
                shutil.copy(str(img_path), str(ep_dir))
            elif label == 1:
                shutil.copy(str(img_path), str(en_dir))
            else:
                raise Exception("error")
