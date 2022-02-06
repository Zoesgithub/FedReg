from FedUtils.models.mnist.cnn import Model
import torch
from functools import partial
from FedUtils.fed.fedprox import FedProx

config = {

    "seed": 1,
    "model": Model,
    "inner_opt": partial(torch.optim.SGD, lr=1e-1, weight_decay=0.0),
    "optimizer": FedProx,
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
    "log_path": "tasks/mnist/FedProx_e40_lr1_g001/train.log",
    "gamma": 0.001,

    "train_transform": None,
    "test_transform": None,
    "eval_train": True


}
