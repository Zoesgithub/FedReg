from torch import nn
from FedUtils.models.utils import Flops
import torch
import sys
import numpy as np
import random


class Attack(object):
    def __init__(self, rep_idx, large_iters, small_iters):
        self.rep_idx = rep_idx  # default word for replacement
        self.large_iters = large_iters  # the number of iters to generate psuedo data
        self.small_iters = small_iters  # the number of iters to generate perturbed data

    def perturb(self, x):
        x = np.copy(x)
        batchsize, length = x.shape

        for idx in range(batchsize):
            replace_idx = random.randint(0, length-1)
            x[idx, replace_idx] = self.rep_idx
        return x

    def generate_adversary(self, x, y):
        large_adv, small_adv = x, x

        for _ in range(self.large_iters):
            large_adv = self.perturb(large_adv)
        for _ in range(self.small_iters):
            small_adv = self.perturb(small_adv)
        return large_adv, small_adv


class Model(nn.Module):
    def __init__(self, num_classes, optimizer=None, learning_rate=None, seed=1, large_iters=40, small_iters=10):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_inp = 80
        torch.manual_seed(123+seed)
        self.embed = nn.Embedding(num_classes+1, 8)  # embed with linear
        self.net = nn.LSTM(input_size=8, hidden_size=256, num_layers=2, batch_first=True)
        self.outnet = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(-1)

        self.size = sys.getsizeof(self.state_dict())
        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters())
        else:
            assert learning_rate, "should provide at least one of optimizer and learning rate"
            self.learning_rate = learning_rate
        self.attack = Attack(self.num_classes, large_iters=large_iters, small_iters=small_iters)

        if torch.cuda.device_count() > 0:
            self.embed = self.embed.cuda()
            self.net = self.net.cuda()
            self.outnet = self.outnet.cuda()

    def set_param(self, state_dict):
        self.load_state_dict(state_dict)
        return True

    def get_param(self):
        return self.state_dict()

    def generate_fake(self, x, y):
        psuedo, perturb = self.attack.generate_adversary(x.long().detach().cpu().numpy(), y.cpu().numpy())
        psuedo, perturb = torch.tensor(psuedo).to(self.embed.weight.device), torch.tensor(perturb).to(self.embed.weight.device)
        psuedo_y, perturb_y = self.predict(psuedo), self.predict(perturb)
        return [psuedo, y, psuedo_y], [perturb, y, perturb_y]

    def loss(self, pred, gt):
        pred = self.softmax(pred)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if len(gt.shape) < len(pred.shape):
            gt = nn.functional.one_hot(gt.long(), self.num_classes).float()
        assert len(gt.shape) == len(pred.shape)
        loss = -gt*torch.log(pred+1e-12)
        loss = loss.sum(-1)
        return loss

    def forward(self, data):
        if data.device != self.embed.weight.device:
            data = data.to(self.embed.weight.device)
        if len(data.shape) == 2:
            data = self.embed(data.long())
        out = self.outnet(self.net(data)[0][:, -1:, :])
        return out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.softmax(self.forward(x))

    def train_onestep(self, data):
        assert self.optimizer, "the optimizer of model should be provided"
        self.train()
        self.zero_grad()
        self.optimizer.zero_grad()
        x, y = data
        pred = self.forward(x)
        loss = self.loss(pred, y).mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def solve_inner(self, data, step_func=None, num_epochs=1):  # step_func should by a closure whose input is (model, data) and output is a callable func to carry out training
        comp = 0.0
        weight = 1.0
        steps = 0
        if step_func is None:
            func = self.train_onestep
        else:
            func = step_func(self, data)

        for _ in range(num_epochs):
            for x, y in data:
                c = func([x, y])
                comp += c
                steps += 1.0
        soln = self.get_param()
        return soln, comp, weight

    def test(self, data):
        tot_correct = 0.0
        loss = 0.0
        self.eval()
        for d in data:
            x, y = d
            with torch.no_grad():
                pred = self.forward(x)
            loss += self.loss(pred, y).sum()
            if len(y.shape) < len(pred.shape):
                y = y.float().squeeze(1)
            else:
                y = y.argmax(-1).float().squeeze(1)
            pred_max = pred.argmax(-1).float().squeeze(1)
            assert len(pred_max.shape) == len(y.shape)
            assert len(y.shape) == 1
            tot_correct += (pred_max == y).float().sum()
        return tot_correct, loss
