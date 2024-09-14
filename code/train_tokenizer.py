#! pip install tokenizers

from pathlib import Path

from  tokenizers import ByteLevelBPETokenizer

paths = "/home/hmahomed/lustre/roberta/wura-xh/train.txt"

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=50_265, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("./")
