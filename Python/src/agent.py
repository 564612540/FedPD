import os

import copy

import torch
from torch import optim
from torch.utils.data import sampler
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from optimizer import PSVRG, FedPD_SGD, FedPD_VR, PSGD

import os.path as osp

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image,dtype=torch.float32), torch.tensor(label)

class Agent:
    def __init__(self, model, args, agent_id, criterion):
        self.local_model = copy.deepcopy(model)
        if args.VR:
            self.local_model_old = copy.deepcopy(model)
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=args.lr,
                                        momentum=args.momentum)
        elif args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=args.lr,
                                         weight_decay=1e-4)
        elif args.optimizer == 'FedProx':
            if args.VR:
                self.optimizer = PSVRG(self.local_model.parameters(), lr=args.lr, mu=args.mu, freq=args.freq_in)
            else:
                self.optimizer = PSGD(self.local_model.parameters(), lr=args.lr, mu=args.mu, freq=args.freq_in)
        elif args.optimizer == 'FedPD':
            if args.VR:
                self.optimizer = FedPD_VR(self.local_model.parameters(), lr=args.lr, mu=args.mu, freq_1=args.freq_in, freq_2 = args.freq_out)
            else:
                self.optimizer = FedPD_SGD(self.local_model.parameters(), lr=args.lr, mu=args.mu, freq=args.freq_in)
        self.id = agent_id
        self.VR = args.VR
        self.opti = args.optimizer
        self.batch_size = args.local_bs
        self.criterion = criterion
        self.use_cuda = args.gpu
        self.init = False

    def train_(self, model, num_its, dataset, idxs, update_model, compute_full):
        sub_data = DatasetSplit(dataset, idxs[self.id])
        loader = DataLoader(sub_data, batch_size=self.batch_size, shuffle=True)

        if self.use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if update_model or self.VR or self.opti != 'FedPD':
            self.local_model.load_state_dict(model.state_dict())
            # print(self.local_model.state_dict()['layer3.0.bias'])
        self.local_model.to(device)
        self.local_model.train()
        if self.VR:
            self.local_model_old.to(device)
        if self.opti == 'FedPD':
            self.optimizer.zero_grad()
            if not self.init:
                # print('first')
                loss = 0.0
                loader_1 = DataLoader(sub_data, batch_size=128, shuffle=True)
                for idx, data in enumerate(loader_1, 0):  # _ start from 0
                    inputs, labels = data
                    if self.use_cuda:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                    outputs = self.local_model(inputs)
                    loss += self.criterion(outputs, labels)
                loss.backward()
                self.init = True
            self.optimizer.zero_grad()
            if compute_full and self.VR:
                loss = 0.0
                loader_1 = DataLoader(sub_data, batch_size=128, shuffle=True)
                for idx, data in enumerate(loader_1, 0):  # _ start from 0
                    inputs, labels = data
                    if self.use_cuda:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                    outputs = self.local_model(inputs)
                    loss += self.criterion(outputs, labels)
                loss.backward()
            self.optimizer.step()
            
        count = 0
        while True:
            for idx, data in enumerate(loader, 0):  # _ start from 0
                inputs, labels = data
                if self.use_cuda:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                if self.VR:
                    self.local_model_old.train()
                    self.local_model_old.zero_grad()
                # forward + backward + optimize
                outputs = self.local_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.VR:
                    outputs_1 = self.local_model_old(inputs)
                    loss_1 = self.criterion(outputs_1, labels)
                    loss_1.backward()
                    if not compute_full or count > 0:
                        dict_old = dict(self.local_model_old.named_parameters())
                        for name, param in self.local_model.named_parameters():
                            param.grad.add(-1, dict_old[name].grad.data)
                    self.local_model_old.load_state_dict(self.local_model.state_dict())

                self.optimizer.step()
                # print(self.local_model.state_dict()['layer3.0.bias'])

                count += 1
                if count == num_its:
                    self.local_model.to('cpu')
                    return self.local_model.state_dict()