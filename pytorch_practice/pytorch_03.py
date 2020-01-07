import torchtext
from torchtext.vocab import Vextors
import torch
import numpy as np
import random


USE_CUDA = torch.cuda().is_avaliable()

random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)

if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
MAX_VOCAB_SIZE = 50000
HIDDEN_SIZE = 128

TEXT = torchtext.data.Field("text8", lower=True)
torchtext.datasets.LanguageModelingDataset.split(path=)


class RNNModel(torch.nn.Module):
    def __init__(self, rnn_type, vocab_size, hidden_size, embed_size):
        super(RNNModel, self).__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def foward(self, text, hidden):
        # text: seq_len * batch
        emb = self.embed(text) # seq_len * batch * embed_size
        output, hidden = self.lstm(emb, hidden)
        # output: seq_len * batch_size * hidden
        # hidden: (1 * batch_size * hidden_size, 1 * batch * hidden_size)
        output_1 = output.view(-1, output.shape[2]) # (seq_len * batch_size) * hidden_size
        out_vocab = self.linear(output_1) # (seq_len * batch_size) * vocab_size
        out_vocab = out_vocab.view(output.shape[0], output.shape[1],out_vocab.shape[-1])
        return out_vocab, hidden

    def init_hidden(self, bsz, require_grad=True):
        weight = next(self.parameters())
        return (
            weight.new_zeros((1, bsz, self.hidden_size), require_grad=require_grad),
            weight.new_zeros((1, bsz, self.hidden_size), require_grad = require_grad)
        )


model = RNNModel(vocab_size=MAX_VOCAB_SIZE, embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE)


def repackage_hidden(h):
    return h.detach()


NUM_EPOCHS = 2
for epoch in range(NUM_EPOCHS):
    model.train()
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.next, batch.target
        output, hidden = model(data, hidden)





















