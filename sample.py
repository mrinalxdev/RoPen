import torch
import torch.nn.functional as F
from config import CONFIG
from model import TinyShakespeareTransformer

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    logits = logits.clone() 

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.tensor(filter_value, device=logits.device), logits)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

@torch.no_grad()
def generate(model, tokenizer, prompt="", max_new_tokens=500, temperature=0.8, top_k=40, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    
    if prompt:
        prompt_tokens = tokenizer.encode(prompt)
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    else:
        newline_token = tokenizer.char_to_idx.get('\n', 0)
        x = torch.tensor([[newline_token]], dtype=torch.long, device=device)
        prompt_tokens = []

    kv_cache = None
    if len(prompt_tokens) > 0:
        _, _, kv_cache = model(x, kv_cache=kv_cache)
        x = x[:, -1:] 

    generated_tokens = prompt_tokens.copy()

    for _ in range(max_new_tokens):
        logits, _, kv_cache = model(x, kv_cache=kv_cache)
        logits = logits[:, -1, :] / temperature
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        x = torch.multinomial(probs, num_samples=1)

        next_token = x.item()
        generated_tokens.append(next_token)

    return tokenizer.decode(generated_tokens)

def load_model_and_generate(checkpoint_path='checkpoint.pt', prompt="", **kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    class SimpleTokenizer:
        def __init__(self, char_to_idx, idx_to_char):
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
        
        def encode(self, text):
            return [self.char_to_idx[ch] for ch in text]
        
        def decode(self, indices):
            return ''.join([self.idx_to_char[i] for i in indices])
    
    tokenizer = SimpleTokenizer(
        checkpoint['tokenizer_char_to_idx'],
        checkpoint['tokenizer_idx_to_char']
    )
    
    model = TinyShakespeareTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model with validation loss: {checkpoint['val_loss']:.4f}")
    
    sample_config = {
        'max_new_tokens': CONFIG['max_new_tokens'],
        'temperature': CONFIG['temperature'],
        'top_k': CONFIG['top_k'],
        'top_p': 0.9,
    }
    sample_config.update(kwargs)
    
    generated = generate(model, tokenizer, prompt, **sample_config)
    return generated

if __name__ == "__main__":
    # prompts = [
    #     "ROMEO:",
    #     "To be or not to be",
    #     "First Citizen:"
    # ]
    
    
        generated = load_model_and_generate(
            prompt="Juliet:",
            max_new_tokens=300,
            temperature=0.9,
            top_k=50
        )
        
        print(generated)