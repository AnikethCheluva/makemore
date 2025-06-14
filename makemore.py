import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np


class Model:
    def __init__(self, inptxt, temp):
        self.words = open(inptxt, 'r').read().splitlines()
        self.stoi = {}
        self.itos = {}
        self.vocabsz = 0
        self.counts = torch.zeros(())
        self.probs = torch.zeros(())
        self.temp = float(temp)
       


    def bigram(self):

        chars = sorted(list(set(''.join(self.words))))

        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocabsz = len(self.stoi)

        self.counts = torch.zeros((self.vocabsz, self.vocabsz), dtype=torch.int32)

        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                self.counts[self.stoi[ch1], self.stoi[ch2]] += 1
    
    def probabilities(self):
        self.probs = (self.counts + self.temp).float()
        self.probs = self.probs / self.probs.sum(dim=1, keepdim=True)
        return self.probs

    def sample(self, numnames, T=1.0):
        self.probabilities()
        out = []

        for _ in range(numnames):
            word = ''
            smp = 0
            while True:
                smp = torch.multinomial(self.probs[smp], 1, replacement=True).item()
                ch = self.itos[smp]
                if ch == '.':
                    break
                word += ch
            out.append(word)

        return out

    def loss(self):
        self.probabilities()
        loglikelihood = 0.0
        numbigrams = 0
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                loglikelihood += torch.log(self.probs[self.stoi[ch1], self.stoi[ch2]])
                numbigrams += 1
        return loglikelihood.item() * -1.0 / numbigrams




m = Model('names.txt', 1)
m.bigram()

print(m.sample(9))
print(m.loss())



