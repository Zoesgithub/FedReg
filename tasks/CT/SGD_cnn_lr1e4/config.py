from FedUtils.models.CT.cnn_parallel import Model
import torch
from functools import partial
from FedUtils.fed.fedavg_sgd import FedAvg
from torchvision import transforms, utils

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.Resize(200),
    transforms.RandomResizedCrop((128,),),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    normalize
])

config = {

    "seed": 1,
    "model": Model,
    "inner_opt": partial(torch.optim.SGD, lr=1e-4, weight_decay=0),
    "optimizer": FedAvg,
    "model_param": (),
    "inp_size": (3*128*128,),
    "train_path": "data/COVID-CT/train/",
    "test_path": ["data/COVID-CT/valid/", "data/COVID-CT/test/"],
    "clients_per_round": 10,
    "num_rounds": 10,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 1,
    "batch_size": 100000,
    "use_fed": 1,
    "log_path": "tasks/CT/SGD_cnn_lr1e4/train.log",
    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": True,



}
