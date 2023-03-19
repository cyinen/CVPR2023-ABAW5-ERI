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
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
from losses.loss import WrappedMSELoss, ClassLoss
from metrics.metrics import mean_pearsons
from .mutil_model import MutilModel

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)
    
    def shared_step(self, batch, stage):
        features, feature_lens, labels, metas = batch
        if stage=='val-ema':
            preds, preds_class = self.model_ema(features)
        else:
            preds, preds_class = self.model(features)
            
        total_loss = self.loss(preds.squeeze(-1), labels.squeeze(-1),feature_lens)
        if self.cri_class and stage=='train':
            l_class = self.cri_class(preds_class, labels)
        else:
            l_class = 0.0
        total_loss += l_class
        if stage=='train':
            self.log(f'{stage}_loss', total_loss,
                    on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': total_loss, 'preds': preds, 'labels': labels}
        
    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, 'train')
        if self.model_ema:
            self.update_model_ema(self.hparams.decay)
        return output
    
    def training_epoch_end(self, outputs):
        self.cacl_mean_pearsons(outputs, 'Train')
        
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.shared_step(batch, 'test')
    
    def test_epoch_end(self, outputs):
        self.cacl_mean_pearsons(outputs, 'Test')
        
    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, 'val')
        out_ema = self.shared_step(batch, 'val-ema')
        return {'val': out, 'val-ema':out_ema}

    def validation_epoch_end(self, outputs):
        out = {'val':[], 'val-ema':[]}
        for o in outputs:
            out['val'].append(o['val'])
            if self.model_ema:
                out['val-ema'].append(o['val-ema'])
        # self.cacu_mean_pearsons(out['val'], 'Val')
        self.cacl_mean_pearsons(out['val-ema'], 'EMA')
        
    def cacl_mean_pearsons(self, outputs, stage):
        preds = [out['preds'].cpu().detach().squeeze().numpy().tolist() for out in outputs]
        labels = [out['labels'].cpu().detach().squeeze().numpy().tolist() for out in outputs]
        score, score_details = mean_pearsons(preds, labels) 
        self.log(f'{stage}_Person', score, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self.shared_step(batch, 'test')
        return output
    
    
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        self.loss = WrappedMSELoss(reduction='mean')
        if self.hparams.cri_weight > 0:
            self.cri_class = ClassLoss(self.hparams.cri_weight)
        else:
            self.cri_class = None

    def load_model(self):
        # name = self.hparams.model_name
        self.model = MutilModel(self.hparams.d_in, 
                                self.hparams.d_model,
                                self.hparams.len_feature)
        self.model_ema = MutilModel(self.hparams.d_in, 
                                self.hparams.d_model,
                                self.hparams.len_feature)
        self.update_model_ema(0)

    def update_model_ema(self, decay=0.99):

        net_g_params = dict(self.model.named_parameters())
        net_g_ema_params = dict(self.model_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)
            
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        # camel_name = ''.join([i.capitalize() for i in name.split('_')])
        # try:
        #     Model = getattr(importlib.import_module(
        #         '.'+name, package=__package__), camel_name)
        # except:
        #     raise ValueError(
        #         f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        # from .mutil_model import MutilModel
        # self.model = self.instancialize(Model)

    # def instancialize(self, Model, **other_args):
    #     """ Instancialize a model using the corresponding parameters
    #         from self.hparams dictionary. You can also input any args
    #         to overwrite the corresponding value in self.hparams.
    #     """
    #     class_args = inspect.getargspec(Model.__init__).args[1:]
    #     print(class_args)
    #     inkeys = self.hparams.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = getattr(self.hparams, arg)
    #     args1.update(other_args)
    #     return Model(**args1)
