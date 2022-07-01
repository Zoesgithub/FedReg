from FedUtils.models.mnist.cnn import Model
from functools import partial
from FedUtils.fed.fedreg import FedReg
from torch.optim import SGD

config = {

    "seed": 1,  # random seed
    "model": partial(Model, learning_rate=1e-1, p_iters=10, ps_eta=2e-1, pt_eta=2e-3),  # the model to be trained
    "inner_opt": None,  # optimizer, in FedReg, only the learning rate is used
    "optimizer": FedReg,  # FL optimizer, can be FedAvg, FedProx, FedCurv or SCAFFOLD
    "model_param": (10,),  # the input of the model, used to initialize the model
    "inp_size": (784,),  # the input shape
    "train_path": "data/mnist_10000/data/train/",  # the path to the train data
    "test_path": "data/mnist_10000/data/valid/",  # the path to the test data
    "clients_per_round": 10,  # number of clients sampled in each round
    "num_rounds": 500,  # number of total rounds
    "eval_every": 1,  # the number of rounds to evaluate the model performance. 1 is recommend here.
    "drop_percent": 0.0,  # the rate to drop a client. 0 is used in our experiments
    "num_epochs": 40,  # the number of epochs in local training stage
    "batch_size": 10,  # the batch size in local training stage
    "use_fed": 1,  # whether use federated learning alrogithms
    "log_path": "tasks/mnist/FedReg/train.log",  # the path to save the log file
    "train_transform": None,  # the preprocessing of train data, please refer to torchvision.transforms
    "test_transform": None,  # the preprocessing of test dasta
    "eval_train": True,  # whether to evaluate the model performance on the training data. Recommend to False when the training dataset is too large
    "gamma": 0.4,  # the value of gamma when FedReg is used, the weight for the proximal term when FedProx is used, or the value of lambda when FedCurv is used


}
