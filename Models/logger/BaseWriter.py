import os
import cv2
import numpy as np
import torchvision.utils as tv_utils

from tensorboardX import SummaryWriter
from utils.modelUtils import findSavedPth, getDate, getTime24, chooseLossFunc, chooseMetric, psnrs

"""
    logger 提供对所有任务和模型提供统一接口，如果后面有需要增加的行为，直接将其变成成员变量

        如果实例化对象时没有传入config参数，那么需要先logging_setup
"""


class BaseWriter():
    def __init__(self, save_log_path = None,
                 writer_extra_str= None,
                 **kargs):
        self.save_log_path    = save_log_path
        self.extra_str = writer_extra_str
        self.experiment_date = kargs["experiment_date"] if kargs.__contains__("experiment_date") else getDate()
        self._writer          = None
        self.tag_prefix      = ""

        self.__bind_writer()

    def __bind_writer(self):
        """
        :param save_log_path:
        :param extr_str: usally tell some info about the model or data setup
        :return:
        """
        save_writer_path = self.save_log_path + '\\' + 'writer'
        os.makedirs(save_writer_path) if not os.path.exists(save_writer_path) else None
        self._writer = SummaryWriter(save_writer_path)
        self.tag_prefix = self.extra_str + "_" + self.experiment_date + '_' + getTime24()+'_'

    def writer_grid_image(self, data_torch, tags, global_step=None, num_per_row=1, normalize=True, scale_each=False):
        """
        This method is not suppose to run to many times
        :param data_torch:
        :param tags:
        :param global_step:
        :param num_per_row:
        :param normalize:
        :param scale_each:
        :return:
        """
        image_grid = tv_utils.make_grid(data_torch, nrow=num_per_row, normalize=normalize, scale_each=scale_each)
        self._writer.add_image( self.tag_prefix+tags, image_grid, global_step)
        print('Write image to writer succeed .')

    def add_scalar(self,tag,value,step,walltime=None):
        self._writer.add_scalar(self.tag_prefix+tag,value,step,walltime=walltime)

    def add_image(self,image, global_step=None,tags=None,cmap="hot"):
        if image.shape[0] == 1 :
            image = cv2.applyColorMap(image[0].astype(np.uint8), cv2.COLORMAP_HOT)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).transpose([2,0,1])
        self._writer.add_image(self.tag_prefix+tags,image, global_step)

