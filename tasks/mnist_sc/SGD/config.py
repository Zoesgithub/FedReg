from FedUtils.models.mnist.cnn import Model
import torch
from functools import partial
from FedUtils.fed.fedavg_sgd import FedAvg

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=1e-1),
    "inner_opt": None,
    "optimizer": FedAvg,
    "model_param": (10,),
    "inp_size": (784,),
    "train_path": "data/mnist_10000_sc/data/train/",
    "test_path": ["data/mnist_10000_sc/data/valid/", "data/mnist_10000_sc/data/test/"],
    "clients_per_round": 10,
    "num_rounds": 500,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 1,
    "batch_size": 10000,
    "use_fed": 1,
    "log_path": "tasks/mnist_sc/SGD/train.log",

    "train_transform": None,
    "test_transform": None,
    "eval_train": True,



}
