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
    "train_path": "data/mnist_10000/data/train/",
    "test_path": ["data/mnist_10000/data/valid/", "data/mnist_10000/data/test/"],
    "clients_per_round": 10,
    "num_rounds": 500,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 40,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/mnist/FedReg_e40_lr1_g4/train.log",
    "train_transform": None,
    "test_transform": None,
    "eval_train": True,
    "gamma": 0.4,
    "eta_s": -2e-1,  # dlr_func,


}
