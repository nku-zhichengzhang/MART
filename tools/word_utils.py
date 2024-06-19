from transformers import BertTokenizer


def initialize_tokenizer(uselarge=False):

    if uselarge:
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        max_input_length = tokenizer.max_model_input_sizes['bert-large-uncased']
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    # print(len(tokenizer.vocab))
    # print(max_input_length)

    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token

    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

    return tokenizer, max_input_length, init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx

