from .server import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch
from torch.optim import SGD


def get_params(model):
    ret = []
    for g in model.optimizer.param_groups:
        for p in g["params"]:
            ret.append(p)
    return ret


class Optim(SGD):
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                param_state = self.state[p]
                assert "c_global" in param_state

                param_state["step"] += 1
                d_p = d_p-param_state["c_local"]+param_state["c_global"]
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])
        return loss


class SCAFFOLD(Server):
    def __init__(self, config, Model, datasets, train_transform, test_transform, traincusdataset, evalcusdataset):
        super(SCAFFOLD, self).__init__(config, Model, datasets, train_transform, test_transform, traincusdataset, evalcusdataset)
        params = get_params(self.model)
        self.c_global = [torch.zeros_like(x) for x in params]
        self.c_local = [[torch.zeros_like(x).cpu() for x in params] for _ in self.clients]

    def set_c(self, clients, indices):
        tstate = [self.c_local[x] for x in indices]
        for c, s in zip(clients, tstate):
            i = 0
            model = c.model
            for g in model.optimizer.param_groups:
                for p in g["params"]:
                    state = model.optimizer.state[p]
                    state["c_global"] = self.c_global[i]
                    state["c_local"] = torch.zeros_like(p)
                    state["c_local"].data.copy_(s[i].cuda())
                    state["step"] = 0
                    i += 1

    def update_c(self, c_locals, deltac_locals, indices):
        tstate = [self.c_local[x] for x in indices]

        for ts, cl, dcl in zip(tstate, c_locals, deltac_locals):
            for t, c, d, g in zip(ts, cl, dcl, self.c_global):
                t.data.copy_(c)
                g.data.add_(d.cuda()/len(self.clients))

    def train(self):
        logger.info("Train with {} workers...".format(self.clients_per_round))
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
            active_clients = np.random.choice(range(len(indices)), round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)
            indices = [indices[x] for x in active_clients]
            active_clients = [selected_clients[x] for x in active_clients]
            csolns = {}
            c_locals = []
            deltac_locals = []
            w = 0
            sopt = self.model.optimizer

            for idx, c in enumerate(active_clients):
                c.set_param(self.get_param())
                self.set_c([c], [indices[idx]])
                soln, stats = c.solve_inner(num_epochs=self.num_epochs)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                    s = []
                    ds = []
                    for g, sg in zip(c.model.optimizer.param_groups, sopt.param_groups):
                        for p, sp in zip(g["params"], sg["params"]):
                            state = c.model.optimizer.state[p]
                            newc = state["c_local"] - state["c_global"] + 1.0 / (g['lr'] * state["step"]) * (sp - p)
                            s.append(newc.detach().cpu())
                            ds.append((newc-state["c_local"]).detach().cpu())
                    deltac_locals.append(ds)
                    c_locals.append(s)

                else:
                    for x in csolns:
                        csolns[x].data.add_(soln[1][x]*soln[0])
                    s = []
                    ds = []
                    for g, sg in zip(c.model.optimizer.param_groups, sopt.param_groups):
                        for p, sp in zip(g["params"], sg["params"]):
                            state = c.model.optimizer.state[p]
                            newc = state["c_local"] - state["c_global"] + 1.0 / (g['lr'] * state["step"]) * (sp - p)
                            s.append(newc.detach().cpu())
                            ds.append((newc-state["c_local"]).detach().cpu())
                    c_locals.append(s)
                    deltac_locals.append(ds)
            self.update_c(c_locals, deltac_locals, indices)
            csolns = [[w, {x: csolns[x]/w for x in csolns}]]
            self.latest_model = self.aggregate(csolns)
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
