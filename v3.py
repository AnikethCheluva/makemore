import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np


class MLP:
    def __init__(self, vocabsz=27, hiddenlayers=128):
        self.vocabsz = vocabsz
        self.hiddenlayers = hiddenlayers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.w1 = torch.randn((vocabsz, hiddenlayers), device=self.device, requires_grad=True)
        self.w2 = torch.randn((hiddenlayers, vocabsz), device=self.device, requires_grad=True)

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.vocabsz).float().to(self.device)
        x = x @ self.w1
        x = F.relu(x)
        logits = x @ self.w2
        return logits


class Model:
    def __init__(self, inptxt, temp):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.words = open(inptxt, 'r').read().splitlines()
        self.stoi = {}
        self.itos = {}
        self.vocabsz = 0
        self.xs = []
        self.ys = []
        self.wei = None
        self.probs = torch.zeros((), device=self.device)
        self.mlp = MLP(27, 256)
        self.lr = 1

        # map characters to ints
        chars = sorted(list(set(''.join(self.words))))
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocabsz = len(self.stoi)

        # Prepare training data and targets
        xs, ys = [], []
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                self.xs.append(self.stoi[ch1])
                self.ys.append(self.stoi[ch2])

        self.xs = torch.tensor(self.xs, device=self.device)
        self.ys = torch.tensor(self.ys, device=self.device)

        # prepare MLP
        self.xencoded = F.one_hot(self.xs, num_classes=27).float().to(self.device)
        self.net = nn.Sequential(
            nn.Linear(27, 128).to(self.device),  # Input layer
            nn.ReLU(),                           # Activation function
            nn.Linear(128, 27).to(self.device)  # Output layer
        )

    def forward(self):
        logits = self.mlp.forward(self.xs)
        self.probs = logits.exp()  # softmax numerator
        self.probs = self.probs / self.probs.sum(dim=1, keepdim=True)  # softmax denominator

    def trainloop(self, iters):
        for i in range(iters):
            self.forward()
            loss = -self.probs[torch.arange(len(self.xs), device=self.device), self.ys].log().mean()
            if i % 100 == 0:
                print(f'Iteration {i}, Loss: {loss.item()}')

            loss.backward()

            # Update weights manually and zero grads
            with torch.no_grad():
                self.mlp.w1 -= self.lr * self.mlp.w1.grad
                self.mlp.w2 -= self.lr * self.mlp.w2.grad

                self.mlp.w1.grad = None
                self.mlp.w2.grad = None

    def sample(self, numnames, T=1.0):
        out = []

        while len(out) < numnames:
            word = ''
            smp = 0
            while True:
                smp = torch.multinomial(self.probs[smp], 1, replacement=True).item()
                ch = self.itos[smp]
                if ch == '.':
                    break
                word += ch
            if 2 < len(word) < 9:
                out.append(word)

        return out


m = Model('names.txt', 1)

print(m.trainloop(10000))
print(m.sample(9))
