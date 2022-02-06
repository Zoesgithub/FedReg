from FedUtils.models.cifar100.resnet9 import Model
import torch
from functools import partial
from FedUtils.fed.fedcurv import FedCurv
from torchvision import transforms, utils

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

config = {

    "seed": 1,
    "model": Model,
    "inner_opt": partial(torch.optim.SGD, lr=5e-2, weight_decay=0),
    "optimizer": FedCurv,
    "model_param": (10,),
    "inp_size": (3*32*32,),
    "train_path": "data/cifar-100-python/data/train/",
    "test_path": ["data/cifar-100-python/data/valid/", "data/cifar-100-python/data/test/"],
    "clients_per_round": 100,
    "num_rounds": 1200,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 10,
    "batch_size": 5,
    "use_fed": 1,
    "log_path": "tasks/cifar100/FedCurv_e10_lr5_g3/train.log",
    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": False,
    "gamma": 1e-3


}
