from FedUtils.models.mnist.cnn import Model
from functools import partial
from FedUtils.fed.fedreg import FedReg
from torch.optim import SGD
config = {

    "seed": 1,
    "model": Model,
    "inner_opt": partial(SGD, lr=1e-1),
    "optimizer": FedReg,
    "model_param": (10,),
    "inp_size": (784,),
    "train_path": "data/mnist_10000_sc/data/train/",
    "test_path": ["data/mnist_10000_sc/data/valid/", "data/mnist_10000_sc/data/test/"],
    "clients_per_round": 10,
    "num_rounds": 500,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 20,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/mnist_sc/FedReg_e20_lr1_g3/train.log",
    "train_transform": None,
    "test_transform": None,
    "eval_train": True,
    "gamma": 0.3,
    "eta_s": -2e-1,  # dlr_func,


}
