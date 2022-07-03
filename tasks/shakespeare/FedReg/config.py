from FedUtils.models.shakespeare.LSTM import Model
from functools import partial
from FedUtils.fed.fedreg import FedReg
config = {
    "seed": 1,
    "model": partial(Model, learning_rate=1, ps_iters=40, pt_iters=0),
    "inner_opt": None,
    "optimizer": FedReg,
    "model_param": (77,),
    "inp_size": (77,),
    "train_path": "data/shakespeare/data/train/",
    "test_path": ["data/shakespeare/data/valid/", "data/shakespeare/data/test/"],
    "clients_per_round": 10,
    "num_rounds": 100,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 5,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/shakespeare/FedReg/train.log",
    "train_transform": None,
    "test_transform": None,
    "eval_train": False,
    "gamma": 1.,

}
