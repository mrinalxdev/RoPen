CONFIG = {
    'vocab_size': None, 
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'max_seq_len': 512,
    'block_size': 128,
    
    # Training
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 1e-1,
    'max_epochs': 50,
    'warmup_steps': 100,
    'max_grad_norm': 1.0,
    
    # Sampling
    'temperature': 0.8,
    'top_k': 40,
    'max_new_tokens': 500,
    
    'device': 'cuda', 
    'compile': False,   
    'save_checkpoint': True,
    'checkpoint_path': 'checkpoint.pt',
    'log_interval': 100,
}