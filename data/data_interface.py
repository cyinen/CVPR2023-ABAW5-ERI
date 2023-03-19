# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from .data_parser import load_data
from .hume_dataset import HumeDataset
class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.data = []
        self.load_data()
        self.setup(kwargs['stage'])

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = HumeDataset(self.data, 'train', self.kwargs['feature'])
            self.valset = HumeDataset(self.data, 'devel', self.kwargs['feature'])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = HumeDataset(
                self.data, 'devel', self.kwargs['feature'])
        if stage == 'predict':
            self.testset = HumeDataset(
                self.data, 'test', self.kwargs['feature'])
        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.
    
        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=2 * self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=2 * self.batch_size, num_workers=self.num_workers, shuffle=False)

    def load_data(self):
        for name in self.kwargs['feature']:
            self.data.append(
                load_data(self.kwargs['task'],
                        self.kwargs['paths'],
                        name,
                        self.kwargs['emo_dim'],
                        self.kwargs['normalize'],
                        self.kwargs['win_len'],
                        self.kwargs['hop_len'],
                        save=self.kwargs['cache'],
                        combined=self.kwargs['combined'],
                        halve_val=self.kwargs['halve_val'])
                )