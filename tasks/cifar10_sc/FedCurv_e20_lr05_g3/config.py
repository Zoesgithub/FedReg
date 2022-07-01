from FedUtils.models.cifar10.resnet9 import Model
import torch
from functools import partial
from FedUtils.fed.fedcurv import FedCurv
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=5e-2),
    "inner_opt": None,
    "optimizer": FedCurv,
    "model_param": (10,),
    "inp_size": (3*32*32,),
    "train_path": "data/cifar-10-batches-py/data/train/",
    "test_path": ["data/cifar-10-batches-py/data/test/", "data/cifar-10-batches-py/data/valid/"],
    "clients_per_round": 100,
    "num_rounds": 240,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 20,
    "batch_size": 5,
    "use_fed": 1,
    "log_path": "tasks/cifar10_sc/FedCurv_e20_lr05_g3/train.log",
    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": False,
    "gamma": 1e-3


}
