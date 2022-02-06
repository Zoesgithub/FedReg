from torch import nn
from FedUtils.models.utils import Flops
import torch
import sys
from .densenet_norm_ import densenet121


class DenseNetModel(nn.Module):
    def __init__(self, optimizer, seed=1):
        super(DenseNetModel, self).__init__()
        self.num_classes = 2
        self.num_inp = 244*244*3
        torch.manual_seed(123+seed)

        self.net = densenet121(num_classes=self.num_classes)  # DenseNet(num_classes=2)#FixupResNet(FixupBasicBlock, [4,4,4,4], 2)
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
        # print(data.shape)
        if len(data.shape) == 2:
            data = data.reshape(-1, 3, 244, 244)
        out = self.net(data)
        # print(out.shape)
        # out=self.softmax(out)
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
