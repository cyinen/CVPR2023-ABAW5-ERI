# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args
import config.config as config
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='EMA_Person',
        mode='max',
        patience=10,
        min_delta=0.0001,
        check_finite=False
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='EMA_Person',
        filename='best-{epoch:02d}-{EMA_Person:.4f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
        
    return callbacks


class PredictionWriter(plc.BasePredictionWriter):

    def __init__(self, output_dir: str, filename: str = 'predict.csv', write_interval: str = 'epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.filename = filename
        os.makedirs(self.output_dir, exist_ok=True)
        
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        print(predictions.keys())
    
    
def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    if load_path is None:
        model = MInterface(**vars(args))
    else:
        print(f"Loading model form {args.load_dir}....")
        model = MInterface(**vars(args))
        args.ckpt_path = load_path
    run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + 'seed_'+str(args.seed)
    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.log_dir,
                               version=run_version)
    callbacks = load_callbacks()
    callbacks.append(PredictionWriter(args.output_dir, args.filename))
    
    args.callbacks = callbacks
    args.logger = logger

    trainer = Trainer.from_argparse_args(args)


    print("args.auto_lr_find=", args.auto_lr_find)

    if 'fit' == args.stage:
        trainer.fit(model, data_module)
    elif 'test' == args.stage:
        model.load_from_checkpoint(checkpoint_path=args.ckpt_path)
        trainer.test(model=model, dataloaders=data_module.test_dataloader())
    elif 'predict' == args.stage:
        trainer.predict(model=model, dataloaders=data_module.test_dataloader())
    else:
        print(f"error! args.stage=={args.stage} not in ['train','test','predict]")

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--task',
                        type=str,
                        default='reaction')
    parser.add_argument('--stage', default='fit', type=str)
    # Basic Training Control
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=104, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str, default='step')
    parser.add_argument('--lr_decay_steps', default=30, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default='', type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='hume_dataset', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--model_name', default='mutil_model', type=str)
    parser.add_argument('--cri_weight', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--log_dir', default='ABAW5', type=str)
    # predict Info
    parser.add_argument('--output_dir', default='results', type=str)
    parser.add_argument('--filename', default='predict.csv', type=str)
    
    # dataset Hyperparameters
    parser.add_argument(
        '--feature',
        nargs='+',
        default=['DeepSpectrum'],
        help="Specify the features used (only one).['ResNet18', 'VGGFace2', 'DeepSpectrum', 'eGeMAPs', 'FAUs']"
    )
    parser.add_argument(
        '--emo_dim',
        default='',
        help='Specify the emotion dimension, only relevant for stress (default: arousal).'
    )
    parser.add_argument(
        '--normalize',
        default=False,
        action='store_true',
        help='Specify whether to normalize features (default: False).')
    parser.add_argument(
        '--win_len',
        type=int,
        default=200,
        help='Specify the window length for segmentation (default: 200 frames).'
    )
    parser.add_argument(
        '--hop_len',
        type=int,
        default=100,
        help='Specify the hop length to for segmentation (default: 100 frames).'
    )
    parser.add_argument('--max_len',
        type=int,
        default=30,
        help='Specify the max feature length for fusion.')
    parser.add_argument(
        '--cache',
        default=True,
        action='store_true',
        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument(
        '--combined',
        default=False,
        action='store_true',
        help='Specify whether to use combined dataset (default: False).')
    # 半个验证集验证是什么意思？
    parser.add_argument(
        '--halve_val',
        default=False,
        action='store_true',
        help='Specify whether to use halved validation set (default: False).')

    # Model Hyperparameters
    parser.add_argument('--ntarget', default=7, type=int)
    parser.add_argument('--len_feature', default=30, type=int)
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--decay', default=0.99, type=float)

    # Other
    parser.add_argument('--save_dir', default='log', type=str)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)

    args = parser.parse_args()

    # List Arguments

    args.paths = {
        'log': os.path.join(config.LOG_FOLDER, args.task),
        'data': os.path.join(config.DATA_FOLDER, args.task),
    }
    args.paths.update({
        'features': config.PATH_TO_FEATURES[args.task],
        'labels': config.PATH_TO_LABELS[args.task],
        'partition': config.PARTITION_FILES[args.task]
    })
    args.d_in = [config.FEATURE_LENGTH[feature] for feature in args.feature]

    main(args)
