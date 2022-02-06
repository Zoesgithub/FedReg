from vision_transformer_pytorch import VisionTransformer
import torch.nn as nn
import torch
import sys
from FedUtils.models.utils import Flops


class Model(nn.Module):
    def __init__(self, num_classes, optimizer, seed=1):
        super(Model, self).__init__()
        self.num_classes = 100
        self.num_inp = 384*384*3
        torch.manual_seed(123+seed)
        self.net = VisionTransformer.from_pretrained("ViT-B_32")
        classifier = self.net.classifier
        s1, s2 = classifier.weight.shape
        s3 = classifier.bias.shape
        self.net.classifier = nn.Linear(s2, self.num_classes)
        self.net.classifier.weight.data.mul_(0).add_(1.0)
        self.net.classifier.bias.data.mul_(0)
        self.size = sys.getsizeof(self.state_dict())
        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if torch.cuda.device_count() > 0:
            self.net = self.net.cuda()
        self.optimizer = optimizer(self.parameters())
        self.softmax = nn.Softmax(-1)

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
        data = data.reshape(-1, 3, 384, 384)
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
