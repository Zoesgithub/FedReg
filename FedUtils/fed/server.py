from torch import nn
import numpy as np
from .client import Client


class Server(object):
    def __init__(self, config, Model, datasets, train_transform=None, test_transform=None, traincusdataset=None, evalcusdataset=None):
        super(Server, self).__init__()
        self.config = config
        self.model_param = config["model_param"]
        self.inner_opt = config["inner_opt"]
        self.clients_per_round = config["clients_per_round"]
        self.num_rounds = config["num_rounds"]
        self.eval_every = config["eval_every"]
        self.batch_size = config["batch_size"]
        self.drop_percent = config["drop_percent"]
        self.num_epochs = config["num_epochs"]
        self.eval_train = config["eval_train"]
        if "gamma" in config:
            self.gamma = config["gamma"]
        else:
            self.gamma = 1.0
        if "eta_s" in config:
            self.eta_s = config["eta_s"]
        if "add_mask" in config:
            self.add_mask = config["add_mask"]
        else:
            self.add_mask = -1
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.model = Model(*self.model_param, self.inner_opt)
        self.cmodel = Model(*self.model_param, self.inner_opt)
        self.traincusdataset = traincusdataset
        self.evalcusdataset = evalcusdataset
        self.clients = self.__set_clients(datasets, Model)

    def __set_clients(self, dataset, Model):
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [(u, g, train_data[u], [td[u] for td in test_data], Model, self.batch_size, self.train_transform, self.test_transform) for u, g in zip(users, groups)]
        return all_clients

    def set_param(self, state_dict):
        self.model.set_param(state_dict)
        return True

    def get_param(self):
        return self.model.get_param()

    def _aggregate(self, wstate_dicts):
        old_params = self.get_param()
        state_dict = {x: 0.0 for x in self.get_param()}
        wtotal = 0.0
        for w, st in wstate_dicts:
            wtotal += w
            for name in state_dict.keys():
                assert name in state_dict
                state_dict[name] += st[name]*w
        state_dict = {x: state_dict[x]/wtotal for x in state_dict}
        return state_dict

    def aggregate(self, wstate_dicts):
        state_dict = self._aggregate(wstate_dicts)
        return self.set_param(state_dict)

    def select_clients(self, seed, num_clients=20):
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(seed)
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        clients = [self.clients[c] for c in indices]
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        return indices, clients

    def save(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        num_samples = []
        tot_correct = []
        clients = [x for x in self.clients if len(x[3][0]['x']) > 0]
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        [m.set_param(self.get_param()) for m in clients]

        for c in clients:
            ct, ns = c.test()
            tot_correct.append(ct)
            num_samples.append(ns)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        num_test = len(tot_correct[0])
        tot_correct = [[a[i] for a in tot_correct] for i in range(num_test)]
        num_samples = [[a[i] for a in num_samples] for i in range(num_test)]
        return ids, groups, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        clients = self.clients
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        [m.set_param(self.get_param()) for m in clients]
        for c in clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        return ids, groups, num_samples, tot_correct, losses
