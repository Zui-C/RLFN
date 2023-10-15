import os
import time
import argparse
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter
from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_crohme_dataset
from models.rlfn import RLFN
from training import train, eval

parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--check', action='store_true')

args = parser.parse_args()



"""config"""
config_file = 'config.yaml'
params = load_config(config_file)

"""seed"""
random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params['device'] = device
print(device)
train_loader, eval_loader = get_crohme_dataset(params)

model = RLFN(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'

print(model.name)
model = model.to(device)

if args.check:
    writer = None
else:
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))
if params['finetune']:
    print(f'finetune path: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])
if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {config_file} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')


min_score, init_epoch = 0, 0
train_word_score = 0
for epoch in range(init_epoch, params['epochs']):
    train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader, writer=writer)
    if epoch >= params['valid_start']:
        eval_loss, eval_word_score, eval_exprate = eval(params, model, epoch, eval_loader, writer=writer)
        print(f'Epoch: {epoch+1} loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')
        if eval_exprate > min_score and not args.check and epoch >= params['save_start']:
            min_score = eval_exprate
            save_checkpoint(model, optimizer, eval_word_score, eval_exprate, epoch+1,
                            optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
