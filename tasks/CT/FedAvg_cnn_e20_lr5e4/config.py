from FedUtils.models.CT.cnn import Model
import torch
from functools import partial
from FedUtils.fed.fedavg import FedAvg
from torchvision import transforms, utils


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize(200),
    transforms.RandomCrop((128,)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    normalize
])
config = {

    "seed": 1,
    "model": partial(Model, learning_rate=5e-4),
    "inner_opt": None,
    "optimizer": FedAvg,
    "model_param": (2,),
    "inp_size": (3*244*244,),
    "train_path": "data/COVID-CT/train/",
    "test_path": ["data/COVID-CT/valid/", "data/COVID-CT/test/"],
    "clients_per_round": 10,
    "num_rounds": 10,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 20,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/CT/FedAvg_cnn_e20_lr5e4/train.log",
    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": True,

}
