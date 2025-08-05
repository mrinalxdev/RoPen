import torch 
import requests
import os
from torch.utils.data import Dataset, DataLoader

def download_shakespeare():
    """Download tiny shakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = 'tinyshakespeare.txt'

    if not os.path.exists(filename):
        print("Downloading tinyshakespeare.txt...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            print(f"Downloaded {len(response.text)} characters from {url}")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Wrote {len(response.text)} characters to {filename}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise

    print(f"Reading {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Read {len(text)} characters from {filename}")
    return text
    


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))

        self.vocab_size=len(chars)
        self.char_to_idx = {ch : i for i, ch in enumerate(chars)}
        self.idx_to_char = {i : ch for i, ch in enumerate(chars)}


    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text)}")
        return [self.char_to_idx[ch] for ch in text]
    

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


class ShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset"""
    def __init__(self, text, tokenizer, block_size):
        self.block_size = block_size
        self.tokenizer = tokenizer
        assert isinstance(text, str), f"Expected str, got {type(text)}"
        assert len(text) > 0, "Text is empty"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list), f"encode() must return list, got {type(encoded)}"
        assert len(encoded) > 0, "Encoded list is empty"
        self.data = torch.tensor(encoded, dtype=torch.long)
        assert self.data.dim() == 1, f"self.data must be 1D, got {self.data.dim()}D"
        print(f"Dataset initialized: {len(self.data)} tokens")

    def __len__(self):
        data_len = len(self.data)  # This will fail if 0D
        result = data_len - self.block_size
        if result < 0:
            raise ValueError(f"Dataset too short: {data_len} tokens, block_size={self.block_size}")
        return result

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

def get_data_loaders(config):
    text = download_shakespeare()
    tokenizer = CharTokenizer(text)
    config['vocab_size'] = tokenizer.vocab_size

    n = len(text)
    print(f"Total text length: {n}")
    split_idx = int(0.9 * n)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"Train text length: {len(train_text)}")
    print(f"Val text length: {len(val_text)}")
    print(f"Block size: {config['block_size']}")

    # 🔴 CRITICAL: Ensure val_text is long enough
    if len(val_text) <= config['block_size']:
        raise ValueError(f"Validation text is too short: {len(val_text)} chars. "
                        f"Must be > block_size ({config['block_size']})")

    train_dataset = ShakespeareDataset(train_text, tokenizer, config['block_size'])
    val_dataset = ShakespeareDataset(val_text, tokenizer, config['block_size'])

    # 🔴 Also validate dataset length
    if len(train_dataset) <= 0:
        raise ValueError(f"Train dataset too short: len(data)={len(train_dataset.data)}, block_size={config['block_size']}")
    if len(val_dataset) <= 0:
        raise ValueError(f"Val dataset too short: len(data)={len(val_dataset.data)}, block_size={config['block_size']}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.device.type == 'cuda' else False
    )

    return train_loader, val_loader, tokenizer
