from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from model import Transformer
from tools import data_process, load_checkpoint, readFilePair, save_checkpoint

device=torch.device("cpu")

# load vocab
bn_vocab_path = Path(__file__).parent / 'bn_vocab.pth'
en_vocab_path = Path(__file__).parent / 'en_vocab.pth'

bn_vocab = torch.load(bn_vocab_path)
en_vocab = torch.load(en_vocab_path)

# data preprocess
data_list = list(readFilePair("./vocab500.bn", 
                              "./vocab500.en"))
src_data=[]
trg_data=[]
for (x, y) in data_list:
    src_data.append(x)
    trg_data.append(y)

X_train, X_test, y_train, y_test = train_test_split(src_data, trg_data, shuffle=True, test_size=0.2)

train_data = data_process(list(zip(X_train, y_train)), bn_vocab, en_vocab)
val_data = data_process(list(zip(X_test, y_test)), bn_vocab, en_vocab)

BATCH_SIZE = 32
PAD_IDX = bn_vocab['<pad>']
BOS_IDX = bn_vocab['<bos>']
EOS_IDX = bn_vocab['<eos>']

def generate_batch(data_batch):
    bn_batch = []
    en_batch = []
    for (bn_item, en_item) in data_batch:
        bn_batch.append(torch.cat([torch.tensor([BOS_IDX]), bn_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    
    bn_batch=pad_sequence(bn_batch, batch_first=True, padding_value=PAD_IDX)
    en_batch=pad_sequence(en_batch, batch_first=True, padding_value=PAD_IDX)
    
    longest_length = max(bn_batch.shape[-1], en_batch.shape[-1])
    
    bn_batch = torch.cat([bn_batch, torch.ones(bn_batch.shape[0], longest_length-bn_batch.shape[-1])*(PAD_IDX)], axis=-1)
    en_batch = torch.cat([en_batch, torch.ones(en_batch.shape[0], longest_length-en_batch.shape[-1])*(PAD_IDX)], axis=-1)
    return bn_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch, drop_last=True)
val_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch, drop_last=True)

# Setting up training phase
load_model = False
save_model = False

# Training hyperparameters
num_epochs = 100
learning_rate = 1e-3
batch_size = BATCH_SIZE

# Model hyperparameters
src_vocab_size = len(bn_vocab)
trg_vocab_size = len(en_vocab)
embedding_size = 8 
num_heads = 4 
num_layers = 4 
dropout = 0.00
max_len = 100 
src_pad_idx = bn_vocab['<pad>']

# Tensorboard
writer = SummaryWriter()
step = 0

model = Transformer(embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_layers, dropout, max_len, device,).to(device)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        torch.nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
           "state_dict": model.state_dict(),
           "optimizer": optimizer.state_dict(),
       }
        save_checkpoint(checkpoint)

    losses = []

    model.train()
    for batch_idx, batch in enumerate(train_iter):
        # Get input and targets and get to cuda
        inp_data = batch[0].to(device)
        target = batch[1].to(device)

        optimizer.zero_grad()

        # Forward prop
        output = model(inp_data, target[:, :])
        output = torch.permute(output, (0, 2, 1))
        
        loss = criterion(output, target.type(torch.long))
        losses.append(loss.item())

        # Back prop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        train_acc = np.sum((target[:,:50]==torch.argmax(output,1)[:,:50]).numpy().astype(int))/batch_size*10

        # plot to tensorboard
        writer.add_scalar("Transformer train loss", loss, global_step=step)
        writer.flush()
        step += 1

        # plot to tensorboard
        writer.add_scalar("Transformer train accuracy", train_acc, global_step=step)
        writer.flush()
        step += 1  

    mean_loss = sum(losses) / len(losses)
    print('Mean loss', mean_loss)
    print('Train accuracy:', train_acc)

    scheduler.step(mean_loss)

    # evaluation
    val_losses = []
    val_acc = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_iter):
            inp_data = batch[0].to(device)
            target = batch[1].to(device)

            # Forward prop
            output = model(inp_data, target[:, :])
            output = torch.permute(output, (0, 2, 1))

            val_loss = criterion(output, target.type(torch.long))
            val_losses.append(val_loss.item())

            val_acc = np.sum((target[:,:50]==torch.argmax(output,1)[:,:50]).numpy().astype(int))/batch_size*10

            # plot to tensorboard
            writer.add_scalar("Transformer validation loss", val_loss, global_step=step)
            writer.flush()
            step += 1

            # plot val accuracy
            writer.add_scalar("Transformer validation accuracy", val_acc, global_step=step)
            writer.flush()
            step += 1

        mean_val_loss = sum(val_losses) / len(val_losses)
        print('Mean val loss', mean_val_loss)
        print('Validation accuracy:', val_acc)

    