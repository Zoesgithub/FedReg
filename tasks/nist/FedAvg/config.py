from FedUtils.models.mnist.cnn import Model
import torch
from functools import partial
from FedUtils.fed.fedavg import FedAvg

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=1e-1),
    "inner_opt": None,
    "optimizer": FedAvg,
    "model_param": (62,),
    "inp_size": (784,),
    "train_path": "data/nist/train/",
    "test_path": ["data/nist/valid/", "data/nist/test/"],
    "clients_per_round": 10,
    "num_rounds": 200,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 10,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/nist/FedAvg/train.log",
    "train_transform": None,
    "test_transform": None,
    "eval_train": False


}
