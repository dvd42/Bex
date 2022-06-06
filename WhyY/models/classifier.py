import numpy as np
from tqdm import tqdm
import torch
from haven import haven_utils as hu
import torchvision
import torch.nn.functional as F
from .backbones import get_backbone


class MLP(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.exp_dict = exp_dict
        # self.model = get_backbone(exp_dict)
        self.model = torch.nn.Sequential(torch.nn.Linear(46, 128, bias=False), torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
                                                         torch.nn.Linear(128, 128, bias=False),
                                                         torch.nn.BatchNorm1d(128),
                                                         torch.nn.ReLU(), torch.nn.Linear(128, 2))
        self.model.cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), betas=(0.9, 0.999),
                                           lr=self.exp_dict["lr"], weight_decay=self.exp_dict["weight_decay"])
        if self.exp_dict["weights"] is not None:
            self.load_state_dict(hu.torch_load(self.exp_dict["weights"]))

        self.generator = get_backbone(exp_dict["generator"]).eval().cuda()
        weights = hu.torch_load(exp_dict["generator"]["weights"])
        self.generator.char_embedding.load_state_dict(weights["char_embedding"])
        self.generator.font_embedding.load_state_dict(weights["font_embedding"])


    def train_on_loader(self, epoch, data_loader):

        self.model.train()
        ret = {}

        for i, batch in enumerate(tqdm(data_loader)):
            res_dict = self.train_on_batch(i, batch)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        return {f"Train/{k}": np.mean(v) for k,v in ret.items()}


    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader):

        self.model.eval()
        ret = {}

        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(tqdm(data_loader)):

            res_dict = self.val_on_batch(epoch, batch_idx, batch)

            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        ret = {f"Val/{k}": np.mean(v) for k,v in ret.items()}
        return ret


    def train_on_batch(self, epoch, batch):

        x, y, categorical_att, continuous_att = batch
        b = x.size(0)

        y = y.cuda()
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()
        with torch.no_grad():
            # z = torch.cat((categorical_att, continuous_att), 1)
            z = self.generator.embed_attributes(categorical_att, continuous_att)

        self.optimizer.zero_grad()

        out = self.model(z)

        loss = F.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()

        acc = (out.argmax(1) == y).sum() / b
        # print(acc)

        return dict(loss=loss.item(), accuracy=float(acc))


    def val_on_batch(self, epoch, batch_idx, batch):

        x, y, categorical_att, continuous_att = batch
        b = x.size(0)

        y = y.cuda()
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()
        z = self.generator.embed_attributes(categorical_att, continuous_att)

        out = self.model(z)

        loss = F.cross_entropy(out, y)

        acc = (out.argmax(1) == y).sum() / b
        # print(acc)

        return dict(loss=loss.item(), accuracy=float(acc))


    def get_state_dict(self):

        ret = {}
        ret["model"] = self.model.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()

        return ret


    def load_state_dict(self, state_dict):

        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])


class ResNet(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()

        self.exp_dict = exp_dict
        self.model = get_backbone(exp_dict)
        self.model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), betas=(0.9, 0.999),
                                           lr=self.exp_dict["lr"], weight_decay=self.exp_dict["weight_decay"])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.exp_dict["max_epoch"])

        if self.exp_dict["weights"] is not None:
            self.load_state_dict(hu.torch_load(self.exp_dict["weights"]))


    def train_on_loader(self, epoch, data_loader):

        self.model.train()
        ret = {}

        for i, batch in enumerate(tqdm(data_loader)):
            res_dict = self.train_on_batch(i, batch)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        self.scheduler.step()
        return {f"Train/{k}": np.mean(v) for k,v in ret.items()}


    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader):

        self.model.eval()
        ret = {}

        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(tqdm(data_loader)):

            res_dict = self.val_on_batch(epoch, batch_idx, batch)

            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        ret = {f"Val/{k}": np.mean(v) for k,v in ret.items()}
        return ret


    def train_on_batch(self, epoch, batch):

        x, y, _, _= batch
        b = x.size(0)

        x = x.cuda()
        y = y.cuda()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = F.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()

        acc = (out.argmax(1) == y).sum() / b

        return dict(loss=loss.item(), accuracy=float(acc))


    def val_on_batch(self, epoch, batch_idx, batch):

        x, y, _, _ = batch
        b = x.size(0)

        x = x.cuda()
        y = y.cuda()

        out = self.model(x)

        loss = F.cross_entropy(out, y)

        acc = (out.argmax(1) == y).sum() / b

        return dict(loss=loss.item(), accuracy=float(acc))


    def get_state_dict(self):

        ret = {}
        ret["model"] = self.model.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()

        return ret


    def load_state_dict(self, state_dict):

        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
