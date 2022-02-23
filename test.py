from distutils.command.build import build
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

def yield_tokens(iter):
    for text in iter:
        yield tokenizer(text)

string = ["Today is not the day that people want to go outside.", "Today is not the day that people want to go outside.", "Today is not the day that people want to go outside.", "Today is not the day that people want to go outside."]
tokenizer = get_tokenizer("basic_english")

