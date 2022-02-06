from torch import nn
from FedUtils.models.utils import Flops
import torch
import sys
from collections import OrderedDict


class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3, inpsize=32):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            #('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            #('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            #('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            #('bn3', torch.nn.BatchNorm2d(4 * width)),
            #('bn3', torch.nn.GroupNorm(1, 4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            #('bn4', torch.nn.GroupNorm(1, 4 * width)),
            #('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            #('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('bn5', torch.nn.GroupNorm(1, 4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            #('bn6', torch.nn.BatchNorm2d(4 * width)),
            #('bn6', torch.nn.GroupNorm(1, 4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv8', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            #('bn8', torch.nn.BatchNorm2d(4 * width)),
            ('bn8', torch.nn.GroupNorm(1, 4 * width)),
            ('relu8', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            #('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            #('pool1', torch.nn.AvgPool2d(3)),
            ('flatten', torch.nn.Flatten()),
            #('linear', torch.nn.Linear(4 * width, num_classes))
            ('linear', torch.nn.Linear((inpsize//3)**2*4 * width, num_classes))
        ]))

    def forward(self, input):
        return self.model(input)


class Model(nn.Module):
    def __init__(self, optimizer, seed=1):
        super(Model, self).__init__()
        self.num_classes = 2
        self.num_inp = 128*128*3
        torch.manual_seed(123+seed)

        self.net = ConvNet(width=128, num_channels=3, num_classes=2, inpsize=128)
        self.size = sys.getsizeof(self.state_dict())
        self.softmax = nn.Softmax(-1)
        self.optimizer = optimizer(params=self.parameters())
        self.iters = 0
        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if torch.cuda.device_count() > 0:
            self.net = self.net.cuda()
        self.net = nn.DataParallel(self.net)

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
        # print(data.shape)
        if len(data.shape) == 2:
            data = data.reshape(-1, 3, 128, 128)
        out = self.net(data)
        # print(out.shape)
        # out=self.softmax(out)
        return out

    def train_onestep(self, data, extra_loss=None):
        self.train()
        self.optimizer.zero_grad()
        x, y = data
        # self.forward(x)
        # self.eval()
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
