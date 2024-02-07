import itertools
import re
import unicodedata
from collections import Counter

import torch
from bntransformer import BanglaTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
"""
=============== Please uncomment necessary lines before training with Bn-En dataset. =================
"""

'''en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
bn_tokenizer = BanglaTokenizer()
counter_bn = Counter()
counter_en = Counter()'''

NON_ENGLISH_PATCH = (
    r'['
    r'^'
    r'a-z'
    r'A-Z'
    r'0-9'
    r'\s'
    r'\u09CD'
    r'\u0021-\u002F'
    r'\u003A-\u0040'
    r'\u005B-\u0060'
    r'\u007B-\u007E'
    r'\u00A0'
    r'\u00A3'
    r'\u00B0'
    r'\u2000-\u2014'
    r'\u2018-\u201D'
    r'\u2028-\u202F'
    r'\u2032-\u2033'
    r'\u2035-\u2036'
    r'\u2060-\u206F'
    r']'
    r'+'
)

NON_BANGLA_PATCH = (
    r'['
    r'^'
    r'\u0981-\u0983'
    r'\u0985-\u098B'
    r'\u098F-\u0990'
    r'\u0993-\u09A8'
    r'\u09AA-\u09B0'
    r'\u09B2'
    r'\u09B6-\u09B9'
    r'\u09BC'
    r'\u09BE-\u09C3'
    r'\u09C7-\u09C8'
    r'\u09CB-\u09CC'
    r'\u09CE'
    r'\u09D7'
    r'\u09DC-\u09DD'
    r'\u09DF'
    r'\u09E6-\u09EF'
    r'\u09F3'
    r'\u0964'
    r'\s'
    r'\u09CD'
    r'\u0021-\u002F'
    r'\u003A-\u0040'
    r'\u005B-\u0060'
    r'\u007B-\u007E'
    r'\u00A0'
    r'\u00A3'
    r'\u00B0'
    r'\u2000-\u2014'
    r'\u2018-\u201D'
    r'\u2028-\u202F'
    r'\u2032-\u2033'
    r'\u2035-\u2036'
    r'\u2060-\u206F'
    r']'
    r'+'
)

def readFilePair(bnFile, enFile) -> zip:
    """
    Read the specified Bangla and English text files in pairs and return a zip object.

    Args:
        bnFile (str): The name of the Bangla text file.
        enFile (str): The name of the English text file.

    Returns:
        zip: A zip object.
    """
    bn = readFileBangla(bnFile)
    en = readFileEnglish(enFile)
    return zip(bn, en)

def readFileBangla(filename) -> list[str]:
    """
    Read the original *.bn file, extract every line, remove leading and trailing whitespace, and return a list of strings.

    Args:
        filename (str): The name of the file to be read.

    Returns:
        List[str]: A list of strings representing the lines extracted from the file.
    """
    with open(filename) as f:
        bnLines=[normalizeBangla(line) for line in f.readlines()]
        return bnLines

def readFileEnglish(filename) -> list[str]:
    """
    Read the original *.en file, extract every line, remove leading and trailing whitespace, and return a list of strings.

    Args:
        filename (str): The name of the file to be read.

    Returns:
        List[str]: A list of strings representing the lines extracted from the file.
    """
    with open(filename) as f:
        enLines = [normalizeEnglish(line) for line in f.readlines()]
        return enLines
    
def unicodeToAscii(s) -> str:
    """
    Convert Unicode string to ASCII by removing diacritics (accent marks).

    Args:
        s (str): The input Unicode string.

    Returns:
        str: The converted ASCII string.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeEnglish(s) -> str:
    """
    Normalize the English text by converting to lowercase, trimming leading and trailing spaces,
    and removing non-letter characters.

    Args:
        s (str): The input English text.

    Returns:
        str: The normalized English text.
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([",.!?])', r" \1 ", s)
    s = re.sub(NON_ENGLISH_PATCH, r" ", s)
    return s

def normalizeBangla(s) -> str:
    """
    Normalize the Bangla text by trimming leading and trailing spaces and removing non-Bengali characters.

    Args:
        s (str): The input Bangla text.

    Returns:
        str: The normalized Bangla text.
    """
    s = s.strip()
    s = re.sub(r'([".,!?।])', r" \1 ", s)
    s = re.sub(NON_BANGLA_PATCH, r" ", s)
    return s

'''def build_vocab(dataForVocab, min_freq=1):
    """
    Builds vocabulary from data and returns the Bangla and English vocabulary objects.

    Args:
        dataForVocab: The data used to build the vocabulary.
        min_freq (int): The minimum frequency of a token to be included in the vocabulary. Default is 1.

    Returns:
        vocab, vocab: A pair of vocab containing the Bangla and English vocabulary objects.
    """
    for item1,item2 in itertools.islice(dataForVocab, len(dataForVocab)):
        counter_bn.update(bn_tokenizer.tokenize(item1))
        counter_en.update(en_tokenizer(item2))

        bn_vocab = vocab(counter_bn, min_freq=min_freq, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        en_vocab = vocab(counter_en, min_freq=min_freq, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

        bn_vocab.set_default_index(bn_vocab['<unk>'])
        en_vocab.set_default_index(en_vocab['<unk>'])
    return bn_vocab, en_vocab'''

'''def data_process(daataa, bn_vocab, en_vocab) -> list:
    """
    Process the data using the provided Bangla and English vocabularies and return a list
    containing the converted tensors.

    Args:
        daataa: The data to be processed.
        bn_vocab: The Bangla vocabulary object.
        en_vocab: The English vocabulary object.

    Returns:
        List: A list containing the processed data as tensors.
    """
    data = []
    for (raw_bn, raw_en) in daataa:
        bn_tensor_ = torch.tensor([bn_vocab[token] for token in bn_tokenizer.tokenize(raw_bn)], dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)
        data.append((bn_tensor_, en_tensor_))
    return data'''

def save_checkpoint(state, filename="my_checkpoint.pth.tar") -> None:
    """
    Save the checkpoint state.

    Args:
        state: The checkpoint state to be saved.
        filename (str): The name of the file to save the checkpoint. Default is "my_checkpoint.pth.tar".

    Returns:
        None
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer) -> None:
    """
    Load the checkpoint into the specified model and optimizer.

    Args:
        checkpoint: The checkpoint to be loaded.
        model: The model object to load the checkpoint's state dictionary into.
        optimizer: The optimizer object to load the checkpoint's optimizer state dictionary into.

    Returns:
        None
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

