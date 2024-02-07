from typing import List

import pytest
import torch
import torch.nn as nn
from torch import optim

from model import Transformer

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

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        torch.nn.init.xavier_uniform_(m.weight)

# Training hyperparameters
num_epochs = 10
learning_rate = 1e-3
batch_size = 32

# Model hyperparameters
src_vocab_size = 10
trg_vocab_size = 10
embedding_size = 8 
num_heads = 4
num_layers = 4
dropout = 0.00
max_len = 100 
src_pad_idx = 1
sequence_len = 10

def test_transformer_model():
    # Creating instance of the model
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
    
    # Applying weights
    model.apply(init_weights)
    # Initializing the optimizer, scheduler & criterion
    optimizer = optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    for i in range(10):
        # Generating test data
        x, y = generate_data_memory(sequence_len, batch_size, key=i)
        inp_data = x.to(torch.device("cpu"))
        target = y.to(torch.device("cpu"))

        # Performing operation
        model.train()
        optimizer.zero_grad()
        output = model(inp_data, target[:, :])
        output = torch.permute(output, (0, 2, 1))
        loss = criterion(output, target.type(torch.long))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # Verifying output
        assert loss.item() >= 0