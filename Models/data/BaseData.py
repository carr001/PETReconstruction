import os
import cv2
import torch
import numpy as np
import torch.utils.data as TorchData

from  data.DataLoader import Dataloader
from  data.DataProcessor import DataProcessor

np.random.seed(0)
torch.random.seed()

class BaseTrainData():
    def __init__(self,config):
        super().__init__()
        """
        If only one dataset, then the data set is assumed to be label(there is input_path in CONFIG);
        if there are two datasets, then train is assumed to be net input, label
        :param config:
        :return:
        """
        if config.__contains__("input_path"):
            self.train_np   = None
            self.label_np   = None
            self.input_path = config["input_path"]
            self.label_path = self.input_path
            self.without_label = True
        else:
            self.train_np = None
            self.label_np = None
            self.train_path    = config["train_path"]
            self.label_path    = config["label_path"]
            self.without_label = False
        # optional
        self.batch_size    = config["batch_size"]   if config.__contains__("batch_size")    else None
        self.shuffle       = config["shuffle"]      if config.__contains__("shuffle")        else False

        self.processor  = DataProcessor()
        self.dataloader = Dataloader()

    def load_train_np(self,train_path=None):
        train_path = train_path if train_path is not None else self.train_path
        return self.dataloader.load_images_from_dir(train_path)

    def load_label_np(self,label_path=None):
        label_path = label_path if label_path is not None else self.label_path
        return self.dataloader.load_images_from_dir(label_path)

    def generate_loader_from_torch(self,input_data_torch,label_data_torch,batch_size=None,shuffle=None):
        batch_size = batch_size                 if batch_size is not None       else self.batch_size
        batch_size = input_data_torch.shape[0]  if self.batch_size==0            else batch_size
        shuffle    = shuffle                    if shuffle is not None          else self.shuffle

        torch_dataset = TorchData.TensorDataset(input_data_torch, label_data_torch)
        loader = TorchData.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        return loader

    def generate_loader_from_np(self,input_data_np,label_data_np,batch_size=None,shuffle=None):
        input_data_torch = torch.from_numpy(input_data_np).float()
        label_data_torch = torch.from_numpy(label_data_np).float()
        return   self.generate_loader_from_torch(input_data_torch,label_data_torch,batch_size,shuffle)

class BaseTestData():
    """
    TestData
    Note:
        dnp: a dictionary with elemens all numpy array
    """
    def __init__(self,config):
        if config.__contains__("input_path"):
            self.test_dnp   = None
            self.label_dnp   = None
            self.input_path = config["input_path"]
            self.label_path = self.input_path
            self.without_label = True
        else:
            self.test_dnp = None
            self.label_dnp = None
            self.test_path    = config["train_path"]
            self.label_path    = config["label_path"]
            self.without_label = False

        self.processor  = DataProcessor()
        self.dataloader = Dataloader()

    def load_test_dnp(self,test_path=None):
        train_path = test_path if test_path is not None else self.test_path
        return self.dataloader.load_images_to_dict(train_path,self.processor.colorspace)

    def load_label_dnp(self,label_path=None):
        label_path = label_path if label_path is not None else self.label_path
        return self.dataloader.load_images_to_dict(label_path,self.processor.colorspace)

#
# class PipeLine():
#     def __init__(self,config):
#         self.pipeline  = []
#         self.parameters= []
#         self.processor = DataProcessor()
#         setattr(self,"colorspace",config["colorspace"])        if config.__contains__("colorspace") else None
#         setattr(self,"patch_size",config["patch_size"])        if config.__contains__("patch_size") else None
#         setattr(self,"stride",config["stride"])                 if config.__contains__("stride")      else None
#         setattr(self,"normalization",config["normalization"]) if config.__contains__("normalization")      else None
#
#     def add_operation(self,operator,kargs):
#         self.pipeline.append(operator)
#         self.parameters.append(kargs)
#
#     def forward(self,data):
#         self.__creat_pipeline()
#
#         x = data
#         for operator,para_karg in (self.pipeline,self.parameters):
#             x = operator(x,**para_karg)
#         return x
#
#     def __creat_pipeline(self):
#         if hasattr(self,"colorspace"):
#             self.add_operation(self.processor.process_colorspace,{"colorspace":getattr(self,"colorspace")})
#
#         if hasattr(self, "augmentation"):
#             pass
#
#         if hasattr(self,"patch_size"):
#             self.add_operation(self.processor.get_patches,{"patch_size":getattr(self,"patch_size"),"stride":getattr(self,"stride")})
#
#         if hasattr(self,"normalization"):
#             self.add_operation(self.processor.normalization_range,{})
