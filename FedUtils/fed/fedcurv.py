from .server import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch
from functools import partial


def step_func(model, data, fed):
    lr = model.learning_rate
    parameters = list(model.parameters())
    flop = model.flop
    fisher, theta_fisher, gamma = fed.fisher, fed.theta_fisher, fed.gamma

    def func(d):
        nonlocal lr, flop, gamma
        model.train()
        model.zero_grad()
        x, y = d
        pred = model.forward(x)
        loss = model.loss(pred, y).mean()
        if fisher is not None:
            for p, f, tf in zip(parameters, fisher, theta_fisher):
                loss += ((p**2*f)*gamma-2*gamma*tf*p).sum()
        grad = torch.autograd.grad(loss, parameters)
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)  # only consider the flop in NN
    return func


class FedCurv(Server):
    def train(self):
        logger.info("Train with {} workers...".format(self.clients_per_round))
        self.fisher = None
        self.theta_fisher = None
        for r in range(self.num_rounds):
            if r % self.eval_every == 0:
                logger.info("-- Log At Round {} --".format(r))
                stats = self.test()
                if self.eval_train:
                    stats_train = self.train_error_and_loss()
                else:
                    stats_train = stats
                logger.info("-- TEST RESULTS --")
                decode_stat(stats)
                logger.info("-- TRAIN RESULTS --")
                decode_stat(stats_train)

            indices, selected_clients = self.select_clients(r, num_clients=self.clients_per_round)
            np.random.seed(r)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)

            csolns = {}
            w = 0
            temp_fisher = None
            temp_theta_fisher = None
            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=partial(step_func, fed=self))  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                else:
                    for x in csolns:
                        csolns[x].data.add_(soln[1][x]*soln[0])
                size = 0.0
                cfisher = None
                ctfisher = None
                for d in c.train_data:
                    x, y = d
                    size += len(x)
                    c.model.eval()
                    gradients = []
                    for i in range(len(x)):
                        loss = c.model.loss(c.model(x[i].unsqueeze(0)), y[i].unsqueeze(0)).squeeze()
                        gradient = torch.autograd.grad(loss, c.model.parameters())
                        with torch.no_grad():
                            gradients.append([_.detach() for _ in gradient])
                    fs = [[a*a for a in x] for x in gradients]
                    fs = [sum([x[i] for x in fs]).detach()*1.0 for i in range(len(fs[0]))]
                    with torch.no_grad():
                        if cfisher is None:
                            cfisher = fs
                            ctfisher = [a*b for a, b in zip(fs, c.model.parameters())]
                        else:
                            cfisher = [a+b for a, b in zip(cfisher, fs)]
                            ctfisher = [a+b*c for a, b, c in zip(ctfisher, fs, c.model.parameters())]
                cfisher = [a.detach()/size for a in cfisher]
                ctfisher = [a.detach()/size for a in ctfisher]
                if temp_fisher is None:
                    temp_fisher = cfisher
                    temp_theta_fisher = ctfisher
                else:
                    temp_fisher = [a+b for a, b in zip(temp_fisher, cfisher)]
                    temp_theta_fisher = [a+b for a, b in zip(temp_theta_fisher, ctfisher)]
                del c
                # csolns.append(soln)
            csolns = [[w, {x: csolns[x]/w for x in csolns}]]

            self.latest_model = self.aggregate(csolns)
            self.fisher = temp_fisher
            self.theta_fisher = temp_theta_fisher
        logger.info("-- Log At Round {} --".format(r))
        stats = self.test()
        if self.eval_train:
            stats_train = self.train_error_and_loss()
        else:
            stats_train = stats
        logger.info("-- TEST RESULTS --")
        decode_stat(stats)
        logger.info("-- TRAIN RESULTS --")
        decode_stat(stats_train)
