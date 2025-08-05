# RoPen
Rope enhanced transformer that pens shakespear, attention-only

### What are we doing in this project 

- training a 6 layer decoder only transformer with RoPe on each attention head, operating on character level tokens from 'tinyshakespear'


### Documenting the workings 

- [x] Day 1 : Datapiple
    - this downloads the tinyshakespeare.txt
    - Character level tokenizer
    - train / val split : `important` Keeping it in the ratio of 90:10 