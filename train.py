import argparse
import pickle
import os
import datetime
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from modules import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=16, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--save-folder', type=str, default='logs', help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='', help='Where to load the trained model if finetunning. Leave empty to train from scratch.')
parser.add_argument('--dims', type=int, default=1, help='The number of input dimensions.')
parser.add_argument('--timesteps', type=int, default=5, help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N', help='Num steps to predict.')
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')

parser.add_argument('--encoder-hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--num-asset', type=int, default=4, help='Number of assets.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#Setting CUDA to False --> For test
args.cuda = False

args.factor = not args.no_factor

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    print("CUDA activated!")
    torch.cuda.manual_seed(args.seed)

if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, 'wb'))
    print("Preparing Log File done")
else:
    print("WARNING : No save_folder provided!")

'''
Data Loader
'''
train_loader = load_data(args.batch_size)

'''
Model Description
'''

encoder = MLPEncoder(args.timesteps * args.dims, 
                    args.encoder_hidden, args.num_asset, args.factor)
decoder = GCNDecoder(1, args.decoder_hidden, 1, dropout=0.5)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

print("BBB")
if args.cuda:
    encoder.cuda()
    decoder.cuda()

print("Everyting Prepared")

def train(epoch, best_val_loss):
    t = time.time()
    print("Current time is : {}".format(t))
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    scheduler.step()
    for batch_idx, (data, blank) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
            blank = blank.cuda()
        data = Variable(data)

        optimizer.zero_grad()
        A = encoder(data)
        output = decoder(data, A)

        target = data[:, :, 1:2]

        loss = F.mse_loss(output, target)

        print(loss.data[0])

        loss.backward()
        optimizer.step()

train(10, 10)