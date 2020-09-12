import torch
from model import SentimentClassifier
import config


def predict_sentiment(sentence):

    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN

    ## TOKENIZER TOKENS
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id

    ## INIT MODEL
    model = SentimentClassifier(
        config.HIDDEN_DIM,
        config.OUTPUT_DIM,
        config.N_LAYERS,
        config.BIDIRECTIONAL,
        config.DROPOUT,
    )

    ## Attach model to device
    model = model.to(config.DEVICE)

    ## Load weights
    model.load_state_dict(
        torch.load(
            "./checkpoints/sentiment_classifier.pt",
            map_location=torch.device(config.DEVICE),
        )
    )

    ## Tokenize input
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[: config.MAX_LEN - 2]
    indexed = (
        [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    )
    tensor = torch.LongTensor(indexed).to(config.DEVICE)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
