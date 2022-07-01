from torch import nn
import numpy as np
from FedUtils.models.utils import Flops, FSGM
import torch
import sys
import torch.nn.functional as F
from fixup.cifar.models.fixup_resnet_cifar import FixupBasicBlock, conv3x3


class FixupLayer(nn.Module):
    """ conv, bias, relu, pool, followed by num_blocks FixupBasicBlocks """

    def __init__(self, in_channels, out_channels, num_blocks, pool):
        super(FixupLayer, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.pool = pool
        self.blocks = nn.Sequential(
            *[FixupBasicBlock(out_channels, out_channels)
              for _ in range(num_blocks)]
        )

    def forward(self, x):
        out = self.conv(x + self.bias1a) * self.scale + self.bias1b
        out = F.relu(out)
        if self.pool is not None:
            out = self.pool(out)
        for block in self.blocks:
            out = block(out)
        return out


class FixupResNet9(nn.Module):
    def __init__(self, channels=None, pool=nn.MaxPool2d(2)):
        super(FixupResNet9, self).__init__()
        self.num_layers = 2
        self.channels = channels or {"prep": 64, "layer1": 128,
                                     "layer2": 256, "layer3": 512}
        self.conv1 = conv3x3(3, self.channels["prep"])
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

        self.layer1 = FixupLayer(self.channels["prep"],
                                 self.channels["layer1"],
                                 1, nn.MaxPool2d(2))
        self.layer2 = FixupLayer(self.channels["layer1"],
                                 self.channels["layer2"],
                                 0, nn.MaxPool2d(2))
        self.layer3 = FixupLayer(self.channels["layer2"],
                                 self.channels["layer3"],
                                 1, nn.MaxPool2d(2))

        self.pool = nn.MaxPool2d(4)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(self.channels["layer3"], 10)

        # initialize conv1
        std = np.sqrt(2 /
                      (self.conv1.weight.shape[0]
                       * np.prod(self.conv1.weight.shape[2:]))
                      )
        nn.init.normal_(self.conv1.weight, mean=0, std=std)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                std = np.sqrt(2 /
                              (m.conv1.weight.shape[0]
                               * np.prod(m.conv1.weight.shape[2:]))
                              ) * self.num_layers ** (-0.5)
                nn.init.normal_(m.conv1.weight, mean=0, std=std)
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, FixupLayer):
                std = np.sqrt(2 /
                              (m.conv.weight.shape[0]
                               * np.prod(m.conv.weight.shape[2:]))
                              )
                nn.init.normal_(m.conv.weight, mean=0, std=std)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def check_params(self):
        std = np.sqrt(2 /
                      (self.conv1.weight.shape[0]
                       * np.prod(self.conv1.weight.shape[2:]))
                      )
        self.conv1.weight.data.clamp_(-std*2, std*2)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                std = np.sqrt(2 /
                              (m.conv1.weight.shape[0]
                               * np.prod(m.conv1.weight.shape[2:]))
                              ) * self.num_layers ** (-0.5)
                m.conv1.weight.data.clamp_(-std*2, std*2)
            elif isinstance(m, FixupLayer):
                std = np.sqrt(2 /
                              (m.conv.weight.shape[0]
                               * np.prod(m.conv.weight.shape[2:]))
                              )
                m.conv.weight.data.clamp_(-std*2, std*2)

    def forward(self, x):
        out = self.conv1(x + self.bias1a) * self.scale + self.bias1b
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).view(out.size()[0], -1)
        out = self.linear(out + self.bias2)
        return out


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 576)


class Model(nn.Module):
    def __init__(self, num_classes, optimizer=None, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_inp = 32*32*3
        torch.manual_seed(123+seed)

        self.net = FixupResNet9()
        self.size = sys.getsizeof(self.state_dict())
        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if torch.cuda.device_count() > 0:
            self.net = self.net.cuda()

        self.softmax = nn.Softmax(-1)

        if optimizer is not None:
            self.optimizer = optimizer(self.parameters())
        else:
            assert learning_rate, "should provide at least one of optimizer and learning rate"
            self.learning_rate = learning_rate

        self.p_iters = p_iters
        self.ps_eta = ps_eta
        self.pt_eta = pt_eta

    def set_param(self, state_dict):
        self.load_state_dict(state_dict)
        return True

    def get_param(self):
        return self.state_dict()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.softmax(self.forward(x))

    def generate_fake(self, x, y):
        self.eval()
        psuedo, perturb = x.detach(), x.detach()
        if psuedo.device != next(self.parameters()).device:
            psuedo = psuedo.to(next(self.parameters()).device)
            perturb = perturb.to(next(self.parameters()).device)
        psuedo = FSGM(self, psuedo, y, self.p_iters, self.ps_eta)
        perturb = FSGM(self, perturb, y, self.p_iters, self.pt_eta)
        psuedo_y, perturb_y = self.predict(psuedo), self.predict(perturb)
        return [psuedo, y, psuedo_y], [perturb, y, perturb_y]

    def loss(self, pred, gt):
        pred = self.softmax(pred)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if len(gt.shape) != len(pred.shape):
            gt = nn.functional.one_hot(gt.long(), self.num_classes).float()
        assert len(gt.shape) == len(pred.shape)
        loss = -gt*torch.log(pred+1e-12)
        loss = loss.sum(1)
        return loss

    def forward(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 3, 32, 32)
        out = self.net(data)
        return out

    def train_onestep(self, data):
        self.train()
        self.zero_grad()
        self.optimizer.zero_grad()
        x, y = data
        pred = self.forward(x)
        loss = self.loss(pred, y).mean()
        loss.backward()
        self.optimizer.step()

        return self.flop*len(x)

    def solve_inner(self, data, num_epochs=1, step_func=None):
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
            pred_max = pred.argmax(-1).float()
            assert len(pred_max.shape) == len(y.shape)
            if pred_max.device != y.device:
                pred_max = pred_max.detach().to(y.device)
            tot_correct += (pred_max == y).float().sum()
        return tot_correct, loss
