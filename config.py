import os
import torch
from transformers import BertTokenizer


HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
BATCH_SIZE = 128
N_EPOCHS = 5

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = TOKENIZER.max_model_input_sizes["bert-base-uncased"]


DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

