from FedUtils.models.utils import CusDataset
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Client(object):
    def __init__(self, id, group, train_data, eval_data, model, batchsize, train_transform=None, test_transform=None, traincusdataset=None, evalcusdataset=None):
        super(Client, self).__init__()
        self.model = model
        self.id = id
        self.group = group
        self.train_samplenum = len(train_data["x"])
        self.num_train_samples = len(train_data["x"])
        self.num_test_samples = [len(ed["x"]) for ed in eval_data]
        drop_last = False
        if traincusdataset:  # load data use costomer's dataset
            self.train_data = DataLoader(traincusdataset(train_data, transform=train_transform), batch_size=batchsize, shuffle=True, drop_last=drop_last)
            self.train_data_fortest = DataLoader(evalcusdataset(train_data, transform=test_transform), batch_size=batchsize, shuffle=False,)
            num_workers = 0
            self.eval_data = [DataLoader(evalcusdataset(ed, transform=test_transform), batch_size=100, shuffle=False, num_workers=num_workers) for ed in eval_data]
        else:
            self.train_data = DataLoader(CusDataset(train_data, transform=train_transform), batch_size=batchsize, shuffle=True, drop_last=drop_last)
            self.train_data_fortest = DataLoader(CusDataset(train_data, transform=test_transform), batch_size=batchsize, shuffle=False)
            self.eval_data = [DataLoader(CusDataset(ed, transform=test_transform), batch_size=100, shuffle=False) for ed in eval_data]
        self.train_iter = iter(self.train_data)

    def set_param(self, state_dict):
        self.model.set_param(state_dict)
        return True

    def get_param(self):
        return self.model.get_param()

    def solve_grad(self):
        bytes_w = self.model.size
        grads, comp = self.model.get_gradients(self.train_data)
        bytes_r = self.model.size
        return ((self.num_train_samples, grads), (bytes_w, comp, bytes_r))

    def solve_inner(self, num_epochs=1, step_func=None):
        bytes_w = self.model.size
        soln, comp, weight = self.model.solve_inner(self.train_data, num_epochs=num_epochs, step_func=step_func)
        bytes_r = self.model.size
        return (self.num_train_samples*weight, soln), (bytes_w, comp, bytes_r)

    def test(self):
        TC = []
        LS = []
        for ed in self.eval_data:
            total_correct, loss = self.model.test(ed)
            TC.append(total_correct)
            LS.append(loss)
        return TC,  self.num_test_samples

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data_fortest)
        return tot_correct, loss, self.train_samplenum
