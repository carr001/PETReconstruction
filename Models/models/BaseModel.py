import os
import torch
from enum import Enum,auto,unique
from utils.modelUtils import findSavedPth,getDate,getTime24,chooseLossFunc,chooseMetric,saveCkpt,loadCkpt

@unique
class Stage(Enum):
    Training = auto()
    Testing = auto()

class BaseModel():
    '''
        BaseModel 负责记录日志,返回模型信息等

        BaseModel不继承nn.Module
    '''
    def __init__(self,config):
        super().__init__()

        self.device          = config['device']             if config.__contains__("device")             else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.save_model_path = config['save_model_path']   if config.__contains__("save_model_path")   else None
        self.pretrain_path   = config['pretrain_path']     if config.__contains__("pretrain_path")      else None
        self.pretrain_epoch  = config['pretrain_epoch']     if config.__contains__("pretrain_epoch")      else None
        self.curr_epoch      = self.pretrain_epoch

        self.stage           = None
        self.experiment_date = getDate()
        self._logger         = None
        self.net             = None
        self.bind_net(**config["net_config"])
        self.bind_logger(**config["logger_config"])

    def bind_logger(self,config=None):
        self._logger = None
        print('bind_logger not implemented')
        raise NotImplementedError

    def bind_net(self,**config):
        """
        Different child class should bing diffent net
        :param config:
        :return:
        """
        self.net = None
        self._logger.print('bind_net not implemented')
        raise NotImplementedError

    def to_device(self,device=None):
        """
        This is called in three situations:
        1.with device parameter device
        2.without device paramter
            either, device is in decided in CONFIG file, or use default device
        :param device:
        :return:
        """
        device = device if device is not None else self.device if hasattr(self,'device') and self.device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._logger.print(f'Using device: {device}')
        return self.net.to(device)

    def get_parameters_list(self):
        return list(self.net.parameters())

    def detach_gpu(self):
        with torch.no_grad():
            parameters = self.get_parameters_list()
            for para in parameters:
                para = para.detach().to("cpu")
                torch.cuda.empty_cache()
                del para

    def curr_device(self):
        return next(self.net.parameters()).device

    def load_pretrained(self,pretrain_path=None,pretrain_epoch=None):
        """
        Load pretraind model, If pretrain_epoch is None, start from self.pretrain_epoch

        The path and date is defined by CONFIG file
        :param pretrain_epoch:
        :return:
        """
        pretrain_epoch = pretrain_epoch if pretrain_epoch is not None else self.pretrain_epoch if self.pretrain_epoch is not None else 0
        if pretrain_epoch is 0:
            self._logger.print(f'Initialize the model with epoch 0')
            return

        pretrain_path  = pretrain_path if pretrain_path is not None else  self.save_model_path +'\\'+ self.pretrain_path
        pth_name = findSavedPth(pretrain_path,pretrain_epoch)
        if pth_name is None:
            self._logger.print(f'Did not find the required saved model, initialize the model with epoch 0')
            self.pretrain_epoch = 0
        else:
            if self.stage == Stage.Training:
                self.net,self.optimizer, self.sched = loadCkpt(pretrain_path + "\\" + pth_name,self.net,self.optimizer,self.sched)
            else:
                self.net, _, _ = loadCkpt(pretrain_path + "\\" + pth_name, self.net)
            self.pretrain_epoch = pretrain_epoch
            self._logger.print(f'Pretrained model loaded with epoch '+str(self.pretrain_epoch))