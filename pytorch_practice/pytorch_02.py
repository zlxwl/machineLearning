# import torch
# import numpy as np
#
# def fizz_buzz_encode(num):
#     if num % 15  == 0:
#         return 3
#     elif num % 5 == 0:
#         return 2
#     elif num % 3 == 0:
#         return 1
#     else:
#         return 0
#
#
# def fizz_buzz_decode(num, prediction):
#     return [str(num), "fizz", "buzz", "fizz_buzz"][prediction]
#
#
# def helper(i):
#     print(fizz_buzz_decode(i, fizz_buzz_encode(i)))
#
#
# for i in range(1, 16):
#     print(helper(i))
#
#
# NUM_DIGITS = 10
# def binary_encode(i, num_digits):
#     return np.array([i >> d & 1 for d in range(num_digits)[::-1]])
#
# trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2** NUM_DIGITS)])
# trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2** NUM_DIGITS)])
#
# print(trX.shape)
# print(trY.shape)
#
# NUM_HIDDEN = 100
# model = torch.nn.Sequential(
#     torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
#     torch.nn.ReLU(),
#     torch.nn.Linear(NUM_HIDDEN, 4)
# )
#
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
#
# BATCH_SIZE = 128
# for epoch in range(200):
#     for start in range(0, len(trX), BATCH_SIZE):
#         end = start + BATCH_SIZE
#         batchX = trX[start:end]
#         batchY = trY[start:end]
#         if torch.cuda.is_available():
#             batchX = batchX.cuda()
#             batchY = batchY.cuda()
#
#         y_pred = model(batchX) # forward
#         loss = loss_fn(y_pred, batchY)
#         print("Epoch:", epoch, loss.item())
#
#         optimizer.zero_grad()
#         loss.backward() # backward
#         optimizer.step() # 计算grad
#
#
# testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])
# with torch.no_grad():
#     testY = model(testX)
#
# predicts = zip(range(1, 101), testY.max(1)[1].data.tolist())
#
# print([fizz_buzz_decode(i, x) for i, x in predicts])
# # print(predicts)
#
#
# ### skip-gram代码
import torch
import torch.nn.functional as F

import math
import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


USE_CUDA = torch.cuda.is_available()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if USE_CUDA:
    torch.cuda.manual_seed(1)

C = 3 # context window
K = 100 # number of negative samples
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 128
LEARNING_RATE = 100
EMBBEDDING_SIZE = 100


def word_tokenizej(text):
    return text.split()


with open("text8", "r", errors="ignore") as f:
    text = f.read()

from collections import Counter
# print(text[:2000].split())
# print(Counter(text[:1000].split()))
vocab = dict(Counter(text.split()).most_common(MAX_VOCAB_SIZE - 1))
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
# print(vocab)

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
print(word_counts.size)
word_freqs = word_counts/np.sum(word_counts)
print(word_freqs[0:100])
word_freqs = word_freqs ** (3.0/4.0)

VOCAB_SIZE = len(idx_to_word)
print(VOCAB_SIZE)


import torch.utils.data as tud
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset,self).__init__()
        self.text_encoded = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)


    def __len__(self):
        # 定义数据集合中有多少个item
        return len(self.text_encoded)

    def __getitem__(self, item):
        center_word = self.text_encoded[item]
        pos_indices = list(range(item-C, item)) + list(range(item+1, item+C+1))
        pos_indices = [i% len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K*pos_words.shape[0], True)
        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(text.split(), word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# print(next(iter(dataloader)))
# for i, (center_word, pos_words, neg_words) in enumerate(dataloader):
#     print(center_word, pos_words, neg_words)

class EmbeddingModel(torch.nn.Module):
    def __init__(self, VOCAB_SIZE, EMBEDDING_SIZE):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = VOCAB_SIZE
        self.embed_size = EMBEDDING_SIZE
        self.in_embed = torch.nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = torch.nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels: {batch_size}
        # pos_labels: {batch_size, {window_size * 2}}
        # neg_labels: {bathc_size, {window_size * 2 * K}}
        input_embedding = self.in_embed(input_labels) # {batch_size, embedding_size}
        pos_embedding = self.in_embed(pos_labels) #{batch_size, {window_size * 2}, embedding_size}
        neg_embedding = self.in_embed(neg_labels) #{bathc_size, {window_size * 2 * K}, embedding_size}

        input_embedding = input_embedding.unsqueeze(2) # {batch_size, embedding_size, 1}
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2) # {batch, window_size * 2
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze(2) # {bathc_size, window_size * 2 * K}

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_neg + log_pos
        return -loss


    def input_embedding(self):
        return self.in_embed.weight.data.cpu().numpy()


model = EmbeddingModel(VOCAB_SIZE=VOCAB_SIZE, EMBEDDING_SIZE=EMBBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
    if USE_CUDA:
        input_labels = input_labels.cuda()
        pos_labels = pos_labels.cuda()
        neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backwards()
        optimizer.step()

        if i%100 == 0:
            print("epoch", e, "iteration", i, loss.item())


















