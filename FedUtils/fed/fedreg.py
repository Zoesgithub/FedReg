from .server import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch
from functools import partial
import copy
import torch.nn as nn


def get_params(model):
    ret = []
    for g in model.optimizer.param_groups:
        for p in g["params"]:
            ret.append(p)
    return ret


def set_params(nmodel, pmodel, rate=10.0):
    nparams = get_params(nmodel)
    pparams = get_params(pmodel)
    for pn, pp in zip(nparams, pparams):
        r = torch.rand(pp.shape).cuda() / rate
        pn.data.mul_(r).add_((1 - r) * pp)


def step_func(model, data, epochs, lr, gamma, eta_s, add_mask):
    def loss_func_nosoft(pred, gt):
        if len(gt.shape) < 2:
            gt = nn.functional.one_hot(gt.long(), pred.shape[-1]).float()
        assert len(gt.shape) == len(pred.shape)
        loss = -gt * torch.log(pred + 1e-30)
        loss = loss.sum(1)
        return loss

    def loss_func(pred, gt):
        pred = nn.Softmax(-1)(pred)
        return loss_func_nosoft(pred, gt)

    def mkfd(d, eta_s, size,  nmodel, reverse=False):
        nmodel.eval()

        fnx, fny = d
        if nmodel.cuda:
            fnx, fny = fnx.cuda(), fny.cuda()
        fd = []
        for _ in range(size):
            fnx.requires_grad = True
            npred = nmodel(fnx)
            if reverse:
                dnx = -torch.autograd.grad(loss_func(npred, (1.0-nn.functional.one_hot(fny.long(), npred.shape[-1]))/(npred.shape[-1]-1.0)).mean(), fnx)[0]
            else:
                dnx = torch.autograd.grad(loss_func(npred, fny).mean(), fnx)[0]
            fnx = (fnx-eta_s*torch.sign(dnx)).detach().clamp(float(fnx.min().detach().cpu().numpy()), float(fnx.max().detach().cpu().numpy()))
            with torch.no_grad():
                ffny = nn.Softmax(-1)(nmodel(fnx))
                fd.append([fnx, fny, ffny.detach()])
        return fd
    comp = 0
    plosses = 0

    now_model = copy.deepcopy(model)
    current_m = copy.deepcopy(model)
    penalty_m = copy.deepcopy(model)

    now_params = get_params(now_model)
    cur_params = get_params(current_m)
    pen_params = get_params(penalty_m)

    params = get_params(model)
    fakedata0 = []
    fakedata1 = []
    for d in data:
        fakedata0 = fakedata0+[mkfd(d, eta_s, 10,  now_model)[-1]]
        fakedata1 = fakedata1+[mkfd(d, eta_s/100.0, 10,  now_model)[-1]]

    tmp = 0
    s2tmp = 0
    fidx = 0
    model.eval()
    now_model.eval()
    current_m.train()
    penalty_m.train()

    for e in range(epochs):

        for i, d in zip(range(len(data)), data):
            x, y = d
            if model.cuda:
                x, y = x.cuda(), y.cuda()
            for _ in range(1):
                fnx, fny0, fny = fakedata0[fidx % len(fakedata0)]
                fnx1, fny1, _ = fakedata1[fidx % len(fakedata0)]
                # estimate running statistics
                '''model.train()
                model(x)
                model.eval()
                s1 = model.state_dict()
                s2 = current_m.state_dict()
                s3 = penalty_m.state_dict()
                for name in s1:
                    if "running" in name:
                        s2[name].data.copy_(s1[name])
                        s3[name].data.copy_(s1[name])
                current_m.load_state_dict(s2)
                penalty_m.load_state_dict(s3)
                current_m.eval()
                penalty_m.eval()'''
                ########
                for p, c, n in zip(params, cur_params, now_params):
                    c.data.copy_(gamma*p+(1-gamma)*n)

                closs = loss_func(current_m(x), y).mean()
                grad1 = torch.autograd.grad(closs, cur_params)

                if add_mask > 0:
                    if len(data) > 1:
                        fnx2, fny20, fny2 = mkfd([x, y], eta_s, 10, now_model)[-1]
                        mask_grad = torch.autograd.grad(loss_func(current_m(fnx2),
                                                                  (1.0-0*nn.functional.one_hot(fny20.long(), fny2.shape[-1]))/(fny2.shape[-1])).mean(), cur_params)
                    else:
                        mask_grad = torch.autograd.grad(loss_func(current_m(fnx),
                                                                  (1.0-0*nn.functional.one_hot(fny0.long(), fny.shape[-1]))/(fny.shape[-1])).mean(), cur_params)
                    s2 = sum([(g2 * g2).sum() for g2 in mask_grad])
                    w = (sum([(g0 * g2).sum() for g0, g2 in zip(grad1, mask_grad)])) / s2.add(1e-30)
                    grad1 = [a-w*b for a, b in zip(grad1, mask_grad)]

                for g1, p in zip(grad1, params):
                    p.data.add_(-lr*g1)
                beta = 0.5
                for c, g1, p, n, pp in zip(cur_params, grad1, params, now_params, pen_params):
                    pp.data.copy_(p*beta+n*(1-beta))
                fidx += 1

                ploss = loss_func(penalty_m(fnx), fny)
                grad2 = torch.autograd.grad(ploss.mean(), pen_params)

                with torch.no_grad():
                    dtheta = [(p-n) for p, n in zip(params, now_params)]

                s2 = sum([(g2*g2).sum() for g2 in grad2])
                w = (sum([(g0*g2).sum() for g0, g2 in zip(dtheta, grad2)]))/s2.add(1e-30)
                w = w.clamp(0.0, )

                ploss_ = loss_func(penalty_m(fnx1), fny1)
                grad3 = torch.autograd.grad(ploss_.mean(), pen_params)
                s3 = sum([(g3*g3).sum() for g3 in grad3])
                w1 = (sum([((g0-w*g2)*g3).sum() for g0, g2, g3 in zip(dtheta, grad2, grad3)]))/s3.add(1e-30)
                w1 = w1.clamp(0.0,)

                tmp += w
                s2tmp += s2

                for g2, g3, p in zip(grad2, grad3, params):
                    p.data.add_(-w*g2-w1*g3)

            comp += model.flop * len(x)
            plosses += closs
    with torch.no_grad():
        model.eval()
        now_model.eval()
        print(plosses / epochs, gamma, tmp/epochs, eta_s, w, loss_func(model(x), y).mean()-loss_func(now_model(x), y).mean(), len(data))
    del now_model
    del current_m
    del penalty_m
    return comp, 1.0


def run_func(client, func, epochs):
    return client.solve_inner(num_epochs=epochs, step_func=func)


class FedReg(Server):
    def train(self):
        logger.info("Train with {} workers...".format(self.clients_per_round))
        epochs = self.num_epochs
        for r in range(self.num_rounds):
            self.round = r

            print(epochs)
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

            self.fakedata = None
            csolns = []

            if isinstance(self.gamma, float):
                gamma = self.gamma
            else:
                gamma = self.gamma(r)

            if isinstance(self.eta_s, float):
                eta_s = self.eta_s
            else:
                eta_s = self.eta_s(r)
            sf = partial(step_func,  gamma=gamma, eta_s=eta_s, add_mask=self.add_mask)
            rf = partial(run_func, epochs=epochs, func=sf)
            w = 0
            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
                soln, stats = rf(c)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                else:
                    for x in csolns:
                        csolns[x].data.add_(soln[1][x]*soln[0])
                del c

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
