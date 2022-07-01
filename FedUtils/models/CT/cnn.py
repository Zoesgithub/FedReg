from torch import nn
from FedUtils.models.utils import Flops, FSGM
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
    def __init__(self, num_classes, optimizer=None, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_inp = 128*128*3
        torch.manual_seed(123+seed)

        self.net = ConvNet(width=128, num_channels=3, num_classes=2, inpsize=128)
        self.size = sys.getsizeof(self.state_dict())
        self.softmax = nn.Softmax(-1)

        if optimizer is not None:
            self.optimizer = optimizer(self.parameters())
        else:
            assert learning_rate, "should provide at least one of optimizer and learning rate"
            self.learning_rate = learning_rate

        self.p_iters = p_iters
        self.ps_eta = ps_eta
        self.pt_eta = pt_eta

        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if torch.cuda.device_count() > 0:
            self.net = self.net.cuda()

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
        data = data.reshape(-1, 3, 128, 128)
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
