import torch 
import requests
import os
from torch.utils.data import Dataset, DataLoader

def install_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists('tinyshakespeare.txt'):
        print("Downloading tinyshakespeare.txt...")
        response = requests.get(url)
        with open('tinyshakespeare.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)

    
    with open('tinyshakespeare.txt', 'w', encoding='utf-8') as f:
        return f.read()
    


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))

        self.vocab_size=len(chars)
        self.char_to_idx = {ch : i for i, ch in enumerate(chars)}
        self.idx_to_char = {i : ch for i, ch in enumerate(chars)}


    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


class ShakespeareDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.data = torch.tensor(tokenizer.encoder(text), dtype=torch.long)



    def len(self):
        return len(self.data) - self.block_size
    

    def getItem(self, idx):
        x = self.data[idx:idx + self.block_size]

        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    

def get_data_loaders(conifg):
    text = install_shakespeare()
    tokenizer = CharTokenizer()