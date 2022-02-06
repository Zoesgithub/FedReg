from torch import nn
import numpy as np
from FedUtils.models.utils import Flops
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
    def __init__(self, num_classes, optimizer, seed=1):
        super(Model, self).__init__()
        self.num_classes = 10
        self.num_inp = 32*32*3
        torch.manual_seed(123+seed)

        self.net = FixupResNet9()
        self.size = sys.getsizeof(self.state_dict())
        self.softmax = nn.Softmax(-1)
        self.optimizer = optimizer(params=self.parameters())
        self.iters = 0
        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if torch.cuda.device_count() > 0:
            self.net = self.net.cuda()

    def set_param(self, state_dict):
        self.load_state_dict(state_dict)
        return True

    def get_param(self):
        return self.state_dict()

    def __loss(self, pred, gt):
        pred = self.softmax(pred)
        if len(gt.shape) < 2:
            gt = nn.functional.one_hot(gt.long(), self.num_classes).float()
        assert len(gt.shape) == len(pred.shape)
        loss = -gt*torch.log(pred+1e-12)
        loss = loss.sum(1)
        return loss

    def forward(self, data):
        data = data.reshape(-1, 3, 32, 32)
        out = self.net(data)
        return out

    def train_onestep(self, data, extra_loss=None):
        self.train()
        self.optimizer.zero_grad()
        x, y = data
        pred = self.forward(x)
        loss = self.__loss(pred, y).mean()
        if not extra_loss is None:
            loss = extra_loss(self, loss, data)
        loss.backward()
        self.optimizer.step()
        self.zero_grad()

        return loss

    def get_gradients(self, data):
        x, y = data
        x = torch.autograd.Variable(x).cuda()
        y = torch.autograd.Variable(y).cuda()
        loss = self.__loss(self.forward(x), y)
        grad = torch.autograd.grad(loss, x)
        flops = self.flop
        return grad, flops

    def solve_inner(self, data, num_epochs=1, extra_loss=None, step_func=None):
        comp = 0.0
        weight = 1.0
        steps = 0
        if step_func:
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                break
            comp, weight = step_func(self, data, num_epochs, lr)
        else:
            for _ in range(num_epochs):
                for x, y in data:
                    self.train_onestep([x, y], extra_loss)
                    comp += self.flop*len(x)
                    steps += 1.0

        soln = self.get_param()
        return soln, comp, weight

    def solve_iters(self, data):
        self.train_onestep(data)
        soln = self.get_param()
        comp = self.flop
        return soln, comp

    def test(self, data):
        tot_correct = 0.0
        loss = 0.0
        self.eval()
        for d in data:
            x, y = d
            with torch.no_grad():
                pred = self.forward(x)
            loss += self.__loss(pred, y).sum()
            pred_max = pred.argmax(-1).float()
            assert len(pred_max.shape) == len(y.shape)
            tot_correct += (pred_max == y).float().sum()
        return tot_correct, loss
