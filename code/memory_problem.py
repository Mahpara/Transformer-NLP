import argparse
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikz
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim

from model import Transformer
from tools import load_checkpoint, save_checkpoint

device=torch.device("cpu")

def generate_data_memory(time_steps: int, n_data: int, n_sequence: int = 10, key: int = 42) -> List:  
    gen = torch.Generator(device='cpu')
    gen.manual_seed(key)
    seq = torch.randint(low=1, high=9, size=(n_data, n_sequence), generator=gen)
    zeros1 = torch.zeros((n_data, time_steps - 1))
    zeros2 = torch.zeros((n_data, time_steps))
    marker = 9 * torch.ones((n_data, 1))
    zeros3 = torch.zeros((n_data, n_sequence))

    x = torch.cat((seq, zeros1, marker, zeros3), dim=1).type(torch.long)
    y = torch.cat((zeros3, zeros2, seq), dim=1).type(torch.long)
    return x, y

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, required=True, help='batch size')
parser.add_argument('-s', '--seq_len', type=int, required=True, help='sequence length')
parser.add_argument('-e', '--epochs', type=int, required=True, help='epochs')

args = parser.parse_args()

# Setting up training phase
load_model = False
save_model = False

# Training hyperparameters
BATCH_SIZE = args.batch_size
num_epochs = args.epochs
learning_rate = 1e-3
batch_size = BATCH_SIZE

# Model hyperparameters
src_vocab_size = 10
trg_vocab_size = 10
embedding_size = 8
num_heads = 4
num_layers = 4
dropout = 0.00
max_len = 100 
src_pad_idx = 1
sequence_len = args.seq_len

# Tensorboard
writer = SummaryWriter()
step = 0

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_layers,
    dropout,
    max_len,
    device,
).to(device)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        torch.nn.init.xavier_uniform_(m.weight)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss()

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# Initialise dictionaries to store the training and validation losses
train_loss_dict = []
val_loss_dict = []
train_acc_dict = []
val_acc_dict = []

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    train_losses = []
    train_acc = []

    model.train()
    for i in range(20):
        x, y = generate_data_memory(sequence_len, batch_size, key=i)
        inp_data = x.to(device)
        target = y.to(device)

        optimizer.zero_grad()

        # Forward prop
        output = model(inp_data, target[:, :])
        output = torch.permute(output, (0, 2, 1))
        loss = criterion(output, target.type(torch.long))
        train_losses.append(loss.item())
        train_loss_dict.append(loss.item())

        # Back prop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        # Gradient descent step
        optimizer.step()
        train_acc = np.sum((y[:,-10:]==torch.argmax(output,1)[:,-10:]).numpy().astype(int))/batch_size*10
        train_acc_dict.append(train_acc)

        # plot train loss
        writer.add_scalar("Memory problem: train loss", loss, global_step=step)
        writer.flush()
        step += 1

        # plot train accuracy
        writer.add_scalar("Memory problem: train accuracy", train_acc, global_step=step)
        writer.flush()
        step += 1

    mean_loss = sum(train_losses) / len(train_losses)
    print('Mean trin loss: {:.2f}'.format(mean_loss))
    scheduler.step(mean_loss)
    print('Train accuracy:', train_acc)

    # evaluation
    val_losses = []
    val_acc = []

    model.eval()
    with torch.no_grad():
        for i in range(10):    
            x, y = generate_data_memory(sequence_len, batch_size, key=i)
            inp_data = x.to(device)
            target = y.to(device)

            # Forward prop
            output = model(inp_data, target[:, :])
            output = torch.permute(output, (0, 2, 1))

            val_loss = criterion(output, target.type(torch.long))
            val_losses.append(val_loss.item())
            val_loss_dict.append(val_loss.item())
            val_acc = np.sum((y[:,-10:]==torch.argmax(output,1)[:,-10:]).numpy().astype(int))/batch_size*10
            val_acc_dict.append(val_acc)
            
            # plot val loss
            writer.add_scalar("Memory problem: validation loss", val_loss, global_step=step)
            writer.flush()
            step += 1

            # plot val accuracy
            writer.add_scalar("Memory problem: validation accuracy", val_acc, global_step=step)
            writer.flush()
            step += 1

        mean_val_loss = sum(val_losses) / len(val_losses)
        print('Mean val loss: {:.2f}'.format(mean_val_loss))
        print('Validation accuracy:', val_acc)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
axes[0][0].plot(train_loss_dict, color='#20bac5', linewidth=2)
axes[0][0].set_ylabel('loss (log)', fontsize=10)
axes[0][0].set_title('Train', fontsize=14)
axes[0][0].set_facecolor('#FDFCFE')
axes[0][0].yaxis.set_ticks_position('both')

axes[0][1].plot(val_loss_dict, color='#20bac5', linewidth=2)
axes[0][1].set_ylabel('loss (log)', fontsize=10)
axes[0][1].set_title('Validation', fontsize=14)
axes[0][1].set_facecolor('#FDFCFE')
axes[0][1].yaxis.tick_right()
axes[0][1].yaxis.set_ticks_position('both')
axes[0][1].yaxis.set_label_position("right")

axes[1][0].plot(train_acc_dict, color='#20bac5', linewidth=2)
axes[1][0].set_xlabel('iterations', fontsize=10)
axes[1][0].set_ylabel('accuracy (%)', fontsize=10)
axes[1][0].set_facecolor('#FDFCFE')
axes[1][0].yaxis.set_ticks_position('both')

axes[1][1].plot(val_acc_dict, color='#20bac5', linewidth=2)
axes[1][1].set_xlabel('iterations', fontsize=10)
axes[1][1].set_ylabel('accuracy (%)', fontsize=10)
axes[1][1].set_facecolor('#FDFCFE')
axes[1][1].yaxis.tick_right()
axes[1][1].yaxis.set_ticks_position('both')
axes[1][1].yaxis.set_label_position("right")

axes[0][0].get_shared_x_axes().join(axes[0][0], axes[1][0])
axes[0][0].set_xticklabels([])

axes[0][1].get_shared_x_axes().join(axes[0][1], axes[1][1])
axes[0][1].set_xticklabels([])

tikz.save(f'batch{batch_size}_embed8_heads4_layers4_seq{sequence_len}.tex', standalone=True)
# Prevent the axis labels from slightly overlapping
#fig.tight_layout()
plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
