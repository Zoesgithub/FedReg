from FedUtils.models.cifar10.resnet9 import Model
from functools import partial
from FedUtils.fed.scaffold import SCAFFOLD, Optim
from torchvision import transforms, utils

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
    "model": Model,
    "inner_opt": partial(Optim, lr=1e-2, weight_decay=0.0),
    "optimizer": SCAFFOLD,
    "model_param": (10,),
    "inp_size": (3*32*32,),
    "train_path": "data/cifar-10-batches-py/data_uni/train/",
    "test_path": ["data/cifar-10-batches-py/data_uni/valid/", "data/cifar-10-batches-py/data_uni/test/"],
    "clients_per_round": 100,
    "num_rounds": 240,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 30,
    "batch_size": 5,
    "use_fed": 1,
    "log_path": "tasks/cifar10/SCAFFOLD_e30_lr01/train.log",
    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": False


}
