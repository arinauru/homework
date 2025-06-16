# Обнаружение аномалий
# Метод главных компонент
# - уменьшим размерность данных
# - восстановим размерность данных

import pandas as pd

df = pd.read_csv("creditcard.csv")
print(df.head())

legit = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

legit = legit.drop(["Class", "Time"], axxis=1)
fraud = fraud.drop(["Class", "Time"], axxis=1)

print(legit.shape)
print(fraud.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=20, random_state=0)
legit_pca = pd.DataFrame(pca.fit_transform(legit), index=legit.index)
fraud_pca = pd.DataFrame(pca.transform(fraud), index=fraud.index)

print(legit_pca.shape)
print(fraud_pca.shape)

legit_restore = pd.DataFrame(pca.inverse_transform(legit_pca), index=legit.index)
fraud_restore = pd.DataFrame(pca.inverse_transform(fraud_pca), index=fraud.index)

print(legit_restore.shape)
print(fraud_restore.shape)

import numpy as np


def anomaly_calc(original, restored):
    loss = np.sum((np.array(original) - np.array(restored)) ** 2, axis=1)
    return pd.Series(data=loss, index=original.index)


legit_calc = anomaly_calc(legit, legit_restore)
fraud_calc = anomaly_calc(fraud, fraud_restore)

import matplotlib.pyplot as plt

fig, ax = plt.subplot(1, 2, sharex="col", sharey="row")
ax[0].plot(legit_calc)
ax[1].plot(fraud_calc)

plt.show()

th = 180

legit_TRUE = legit_calc[legit_calc < th].count()
legit_FALSE = legit_calc[legit_calc >= th].count()

fraud_TRUE = fraud_calc[fraud_calc >= th].count()
fraud_FALSE = fraud_calc[fraud_calc < th].count()

print(legit_TRUE)
print(fraud_TRUE)

print(legit_FALSE)
print(fraud_FALSE)

# Обработка естественного языка NLP
# Языковая модель позволяет предсказать следующее слово зная предыдущее. Метки не требуются, но нужно очень много текста.
# Метки получаются автоматически из данных

# https://www.fast.ai/

from fastai.text.all import *

path = untar_data(URLs.HUMAN_NUMBERS)

print(path.ls())

lines = L()
with open("data.txt") as f:
    lines += L(*f.readlines())

text = " ".join([l.strip() for l in lines])
# print(text[:50])

tokens = text.split(" ")
# print(tokens[:10])

vocab = L(*tokens).unique()
# print(vocab[:10])

word2index = {w: i for i, w in enumerate(vocab)}

nums = L(word2index[i] for i in tokens)

# 1. Список из всех последовталеьностей из трех слов
seq = L((tokens[i:i + 3], tokens[i + 3]) for i in range(0, len(tokens) - 4, 3))
print(seq[:10])

seq = L((nums[i:i + 3], nums[i + 3]) for i in range(0, len(nums) - 4, 3))
print(seq[:10])

seq = L((tensor(nums[i:i + 3]), nums[i + 3]) for i in range(0, len(nums) - 4, 3))
print(seq[:10])

bs = 64
cut = int(len(seq) * 0.8)
dls = DataLoaders.from_dsets(seq[:cut], seq[cut:], bs=bs, shuffle=False)


class Model1(Module):
    def __int__(self, vocab_sz, n_hidden):
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden, vocab_sz)

    def forward(self, x):
        h = F.relu(self.h_h(self.i_h(x[:, 0])))
        h = h + self.i_h(x[:, 1])
        h = F.relu(self.h_h(h))
        h = h + self.i_h(x[:, 2])
        h = F.relu(self.h_h(h))
        return self.h_o(h)


learn = Learner(dls, Model1(len(vocab), bs), loss=F.cross_entropy, metrics=accuracy)

learn.fit_one_cycle(4, 0.001)

n = 0
counts = torch.zeros(len(vocab))

for x, y in dls.valid:
    n += y.shape[0]
    for i in range_of(vocab):
        counts[i] += (y == 1).long().sum()

print(counts)

index = torch.argmax(counts)

print(index, vocab[index.item()], count[index].item() / n)
