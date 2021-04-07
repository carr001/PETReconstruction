import os
import logging
import torchvision.utils as tv_utils

from tensorboardX import SummaryWriter
from utils.modelUtils import findSavedPth,getDate,getTime24,chooseLossFunc,chooseMetric,psnrs

"""
    logger 提供对所有任务和模型提供统一接口，如果后面有需要增加的行为，直接将其变成成员变量
    
        如果实例化对象时没有传入config参数，那么需要先logging_setup
"""
class BaseLogging():
    def __init__(self,save_log_path=None,
                 log_name=None,
                 **kargs):
        self.save_log_path   = save_log_path
        self.log_name        = log_name

        self.experiment_date = kargs["experiment_date"] if kargs.__contains__("experiment_date") else getDate()
        self.__bind_logging()

    def __bind_logging(self):

        os.makedirs(self.save_log_path ) if not os.path.exists(self.save_log_path ) else []
        log_path = self.save_log_path + '\\' + self.experiment_date  + '.log'
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s: %(message)s')

    def print(self,arg):
        print(arg)
        logging.info(arg)

