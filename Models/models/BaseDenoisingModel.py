import os
import torch
import numpy as np
import torch.optim as optim

from enum import Enum,auto,unique
from models.BaseModel import BaseModel
from logger.DenoisingLogger import DenoisingLogger
from trainer.DenoisingTrainer import DenoisingTrainer
from tester.DenoisingTester import DenoisingTester
from utils.modelUtils import findSavedPth,getDate,getTime24,chooseLossFunc,chooseOptimizer,chooseMetric,chooseInitializer,chooseSched,saveCkpt,loadCkpt

@unique
class Stage(Enum):
    Training = auto()
    Testing = auto()
class BaseTrainModel(BaseModel):
    """
     BaseTrainModel  保存模型，加载模型，训练模型，
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.stage       = Stage.Training
        self.end_epoch   = config['end_epoch']

        self.optimizer   = chooseOptimizer(self.net,**config['optimizer_config'])
        self.sched       = chooseSched(self.optimizer,**config['sched_config'])
        self.loss_func   = chooseLossFunc(config['loss_func_name'])
        self.initializer = chooseInitializer(config['initializer_name']) if config.__contains__("initializer_name") else None
        self.extra_str   = self._logger.writer.extra_str

        self.bind_trainer(**config["trainer_config"])
        self.train_init()

    def bind_logger(self,**config):
        self._logger = DenoisingLogger(**config)

    def bind_trainer(self,**config):
        self.trainer = DenoisingTrainer(**config)

    def train_init(self,pretrain_epoch=None,device=None):
        self.to_device(device)
        self.net_initialize(self.initializer)
        self.load_pretrained(pretrain_epoch)

    def net_initialize(self,initializer):
        self.net.apply(initializer)if initializer is not None else None

    def train(self,dataset):
        self.trainer.train(self,dataset)

    def save_this_model(self):
        """
        save model on today's dir
        :return:
        """
        device = self.curr_device()
        self.net.to('cpu')
        self._logger.print(f'Saving model for epoch {self.curr_epoch}')
        path = self.save_model_path+"\\"+self.experiment_date
        os.makedirs(path) if not os.path.exists(path) else None # None means do nothing
        saveCkpt( path+"\\"+self.extra_str+f'_epoch{self.curr_epoch}'+'.ckpt',self.net,self.optimizer,self.sched)
        self.net.to(device)
        self._logger.print(f'Model saved for epoch {self.curr_epoch}')

class BaseTestModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)
        self.stage       = Stage.Testing

        self.bind_tester(**config["tester_config"])
        self.pretrain_epoch = config['pretrain_epoch'] if config.__contains__("pretrain_epoch") else None
        self.load_pretrained()

    def bind_data(self,data):
        self.data = data

    def bind_logger(self,**config):
        self._logger = DenoisingLogger(**config)

    def bind_tester(self,**config):
        self.tester = DenoisingTester(self,**config)



