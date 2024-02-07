import torch
from tools import readFilePair, build_vocab
from pathlib import Path

#building vocab
dataForVocab = list(readFilePair("./corpus.bn",
                                  "./corpus.en"))
bn_vocab, en_vocab = build_vocab(dataForVocab)

bn_vocab_path = Path(__file__).parent / 'bn_vocab.pth'
en_vocab_path = Path(__file__).parent / 'en_vocab.pth'

torch.save(bn_vocab, bn_vocab_path)
torch.save(en_vocab, en_vocab_path)

print(f'Bangla vocab length:{len(bn_vocab)} \n English vocab length:{len(en_vocab)}')

