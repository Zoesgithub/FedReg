from FedUtils.models.emnist.cnn import Model
from functools import partial
from FedUtils.fed.fedreg import FedReg
from torch.optim import SGD

config = {

    "seed": 1,
    "model": Model,
    "inner_opt": partial(SGD, lr=2.0e-1),
    "optimizer": FedReg,
    "model_param": (10,),
    "inp_size": (784,),
    "train_path": "data/emnist/data_10000/train/",
    "test_path": ["data/emnist/data_10000/valid/", "data/emnist/data_10000/test/"],
    "clients_per_round": 20,
    "num_rounds": 500,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 20,
    "batch_size": 24,
    "use_fed": 1,
    "log_path": "tasks/emnist/FedReg_e20_lr2_g4/train.log",
    "train_transform": None,
    "test_transform": None,
    "eval_train": False,
    "gamma": 0.4,
    "eta_s": -5e-2,


}
