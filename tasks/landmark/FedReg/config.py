from FedUtils.models.landmark.densenet import DenseNetModel
import torch
from functools import partial
from FedUtils.fed.fedreg import FedReg
from torchvision import transforms, utils
normalize = transforms.Normalize(mean=[0.4852, 0.4936, 0.4863], std=[0.2540, 0.2565, 0.2917])
transform_train = transforms.Compose([
    transforms.Resize(72),
    transforms.RandomResizedCrop((64,)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize
])

config = {

    "seed": 1,
    "model": partial(DenseNetModel, learning_rate=1e-1, p_iters=10, ps_eta=1e-1, pt_eta=1e-3),
    "inner_opt": None,
    "optimizer": FedReg,
    "model_param": (2028,),
    "inp_size": (3*64*64,),
    "train_path": "data/landmarks/train/",
    "test_path": ["data/landmarks/valid/", "data/landmarks/test/"],
    "image_path": "./data/landmarks/summary.hdf5",
    "clients_per_round": 100,
    "num_rounds": 1000000,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 40,
    "batch_size": 30,
    "use_fed": 1,
    "log_path": "tasks/landmark/FedReg/train.log",
    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": False,
    "gamma": 0.5,



}
