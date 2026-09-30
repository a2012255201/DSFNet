import os

import torch
from engine.dehaze import train
from data.uieb import UIEBTrain, UIEBValid
from torch.utils.data import DataLoader
from torch.optim import AdamW

from utils.common_utils import parse_yaml, Logger, save_dict_as_yaml
from torch.cuda.amp import GradScaler

from model.arch.DSFNet import DSFNet

def configuration_dataloader(hparams, stage_index):
    train_dataset = UIEBTrain(
        folder=hparams['data']['train_path'],
        size=hparams['data']['train_img_size'][stage_index]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams['data']['train_batch_size'][stage_index],
        shuffle=True,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    valid_dataset = UIEBValid(
        folder=hparams['data']['valid_path'],
        size=256

    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    return train_loader, valid_loader

def configuration_optimizer(model, hparams):
    optimizer = AdamW(
        params=model.parameters(),
        lr=hparams['optim']['lr_init'],
        weight_decay=hparams['optim']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=3)
    return optimizer, scheduler

if __name__ == '__main__':
    args = parse_yaml(r'./config.yaml')
    model = DSFNet().cuda()
    scaler = GradScaler()
    base_path = os.path.join(args['train']['save_dir'],
                                 args['train']['model_name'],
                                 args['train']['task_name'])
    logger = Logger(os.path.join(base_path, 'tensorboard'))
    optimizer, scheduler = configuration_optimizer(model, args)
    save_dict_as_yaml(args, base_path)

    for i in range(len(args['train']['stage_epochs'])):
        print('\nStage {} Start'.format(i + 1))
        if i != 0:
            args['train']['resume'] = True
        train_loader, valid_loader = configuration_dataloader(args, i)
        train(args, model, optimizer, scaler, scheduler, logger, train_loader, valid_loader, i)
        print('Stage {} End'.format(i + 1))
