import numpy as np
import argparse
import importlib
import random
import torch
from FedUtils.models.utils import read_data, CusDataset, ImageDataset
from torch.utils.data import DataLoader
from loguru import logger
from functools import partial
import os
torch.backends.cudnn.deterministic = True


def allocate_memory():
    total, used = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    ).read().split('\n')[0].split(',')
    total = int(total)
    total = int(total * 0.7)
    n = torch.cuda.device_count()
    for _ in range(n):
        x = torch.rand((256, 1024, total)).cuda(_)
        del x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The config file")
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace("/", "."))
    config = config.config
    logger.add(config["log_path"])

    random.seed(1+config["seed"])
    np.random.seed(12+config["seed"])
    torch.manual_seed(123+config["seed"])
    torch.cuda.manual_seed(123+config["seed"])

    Model = config["model"]
    inner_opt = config["inner_opt"]
    if "landmarks" in config["train_path"]:  # load landmark data
        assert "image_path" in config
        Dataset = partial(ImageDataset, image_path=config["image_path"])
        clients, groups, train_data, eval_data = read_data(config["train_path"], config["test_path"])
    else:  # load other data
        clients, groups, train_data, eval_data = read_data(config["train_path"], config["test_path"])
        Dataset = CusDataset

    if config["use_fed"]:
        Optimizer = config["optimizer"]
        t = Optimizer(config, Model, [clients, groups, train_data, eval_data], train_transform=config["train_transform"],
                      test_transform=config['test_transform'], traincusdataset=Dataset, evalcusdataset=Dataset)
        t.train()
    else:
        train_data_total = {"x": [], "y": []}
        eval_data_total = {"x": [], "y": []}
        for t in train_data:
            train_data_total["x"].extend(train_data[t]["x"])
            train_data_total["y"].extend(train_data[t]["y"])
        for t in eval_data:
            eval_data_total["x"].extend(eval_data[t]["x"])
            eval_data_total["y"].extend(eval_data[t]["y"])
        train_data_size = len(train_data_total["x"])
        eval_data_size = len(eval_data_total["x"])
        train_data_total_fortest = DataLoader(Dataset(train_data_total, config["test_transform"]), batch_size=config["batch_size"], shuffle=False,)
        train_data_total = DataLoader(Dataset(train_data_total, config["train_transform"]), batch_size=config["batch_size"], shuffle=True, )
        eval_data_total = DataLoader(Dataset(eval_data_total, config["test_transform"]), batch_size=config["batch_size"], shuffle=False,)
        model = Model(*config["model_param"], optimizer=inner_opt)
        for r in range(config["num_rounds"]):
            model.solve_inner(train_data_total)
            stats = model.test(eval_data_total)
            train_stats = model.test(train_data_total_fortest)
            logger.info("-- Log At Round {} --".format(r))
            logger.info("-- TEST RESULTS --")
            logger.info("Accuracy: {}".format(stats[0]*1.0/eval_data_size))
            logger.info("-- TRAIN RESULTS --")
            logger.info(
                "Accuracy: {} Loss: {}".format(train_stats[0]/train_data_size, train_stats[1]/train_data_size))


if __name__ == "__main__":
    main()
