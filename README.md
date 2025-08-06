# RoPen
Rope enhanced transformer that pens shakespear, attention-only

### What are we doing in this project 

- training a 6 layer decoder only transformer with RoPe on each attention head, operating on character level tokens from 'tinyshakespear'


### Documenting the workings 

- [x] Day 1 : Datapiple
    - this downloads the tinyshakespeare.txt
    - Character level tokenizer
    - train / val split : `important` Keeping it in the ratio of 90:10 

- [x] Day 2 : Model Configuration and pure attention transformer implmentation
    - RoPe attention with tied Q/K projections
    - Pure attention block (no FFN layers)
    - Weight trying between input/output embeddings

- [x] Wrap up, Complete training pipeline
    - AdamW optimizer with weight decay
    - Cosine learning rate schedule with warmup
    - Gradient clipping
    - Validation loss tracking
    - Automatic checkpointing

- [x] Text generation part
    - Top-k and top-p (nucleus) sampling
    - Temperature control
    - Checkpoint loading
    - Multiple prompt examples


Lets add a feed forward network to transformer block also teach and implement AliBi also add kv cache for faster generation, currently it re-computes all activation every step


later : 

improve rope : support partial head dimensions, fix : handle odd dimensions,  Add Residual Stream Dropout & Stochastic Depth
✅ Add Dropout in Forward Pass