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
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.tensor([[tokenizer.char_to_idx.get('\n', 0)]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature 

        logits_1d = logits.squeeze(0)  
        logits_1d = top_k_top_p_filtering(logits_1d, top_k=top_k, top_p=top_p)
        logits = logits_1d.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return tokenizer.decode(idx[0].tolist())

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
            temperature=0.9,  # Higher = more creative
            top_k=50
        )
        
        print(generated)