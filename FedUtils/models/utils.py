from thop import profile
import os
import json
from torch.utils.data import TensorDataset
import numpy as np
import torch
from loguru import logger
from PIL import Image
import h5py


def FSGM(model, inp, label, iters, eta):
    inp.requires_grad = True
    minv, maxv = float(inp.min().detach().cpu().numpy()), float(inp.max().detach().cpu().numpy())
    for _ in range(iters):
        loss = model.loss(model.forward(inp), label).mean()
        dp = torch.sign(torch.autograd.grad(loss, inp)[0])
        inp.data.add_(eta*dp.detach()).clamp(minv, maxv)
    return inp


class CusDataset(TensorDataset):
    def __init__(self, data, transform=None):
        assert "x" in data
        assert "y" in data
        self.data = {}
        self.data["x"] = (data["x"])
        self.data["y"] = (data["y"])
        self.transform = transform

    def __getitem__(self, item):
        if self.transform is None:
            ret = torch.tensor(self.data['x'][item])
        else:
            ret = np.array(self.data["x"][item]).astype("uint8")
            if ret.shape[-1] == 3:
                ret = ret
            elif ret.shape[0] == 3:
                ret = ret.transpose(1, 2, 0)
            else:
                ret = ret
            ret = self.transform(Image.fromarray(ret))

        return [ret, torch.tensor(self.data["y"][item])]

    def __len__(self):
        return len(self.data["x"])


class ImageDataset(TensorDataset):
    def __init__(self,  data, transform=None, image_path=None):
        self.transform = transform

        assert "x" in data
        assert "y" in data
        self.data = {}
        self.data["x"] = (data["x"])
        self.data["y"] = (data["y"])
        if len(self.data["x"]) < 20000:
            File = h5py.File(image_path, "r")
            self.image_path = {}
            for name in self.data["x"]:
                name = name.replace(".png", "")
                self.image_path[name+"_X"] = np.array(File[name+"_X"])
                self.image_path[name+"_Y"] = np.array(File[name+"_Y"])
            File.close()
        else:
            self.image_path = h5py.File(image_path, "r")

    def __getitem__(self, item):
        path = self.data["x"][item]
        path = path.replace(".png", "")
        image, y = Image.fromarray((np.array(self.image_path[path+"_X"])*255).transpose(1, 2, 0).astype(np.uint8)), self.image_path[path+"_Y"]
        if self.transform is None:
            ret = torch.tensor(image)
        else:
            try:
                assert image.mode == "RGB"
            except:
                image = image.convert("RGB")
            ret = self.transform(image)

        return [ret, torch.tensor(self.data["y"][item])]

    def __len__(self):
        return len(self.data["x"])


def Flops(model, inp):
    return profile(model, inputs=(inp,), verbose=False)[0]


def read_data(train_data_path, test_data_path):
    if not isinstance(test_data_path, list):
        test_data_path = [test_data_path, ]
    groups = []
    train_data = {}
    test_data = [{} for _ in test_data_path]
    train_files = os.listdir(train_data_path)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_path, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])
    for F, td in zip(test_data_path, test_data):
        test_files = os.listdir(F)
        test_files = [f for f in test_files if f.endswith(".json")]
        for f in test_files:
            file_path = os.path.join(F, f)
            with open(file_path, "r") as inf:
                cdata = json.load(inf)
            td.update(cdata["user_data"])
    clients = list(sorted(train_data.keys()))
    return clients, groups, train_data, test_data


def decode_stat(stat):
    if len(stat) == 4:
        ids, groups, num_samples, tot_correct = stat
        if isinstance(num_samples[0], list):
            assert len(num_samples) == len(tot_correct)
            idx = 0
            for a, b in zip(tot_correct, num_samples):
                logger.info("Test_{} Accuracy: {}".format(idx, sum(a) * 1.0 / sum(b)))
                idx += 1
        else:
            logger.info("Accuracy: {}".format(sum(tot_correct) / sum(num_samples)))
    elif len(stat) == 5:
        ids, groups, num_samples, tot_correct, losses = stat
        logger.info("Accuracy: {} Loss: {}".format(sum(tot_correct) / sum(num_samples), sum(losses) / sum(num_samples)))
    else:
        raise ValueError
