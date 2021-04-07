import os
import sys

import numpy as np
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__),'../../../Models/utils')))


from data.BaseData import BaseTrainData,BaseTestData

class DenoisingTrainData(BaseTrainData):
    def __init__(self,config=None):
        super().__init__(config)
        self.gauss_sigma   = config["gauss_sigma"]  if config.__contains__("gauss_sigma")   else 1
        self.colorspace    = config["colorspace"]  if config.__contains__("colorspace")   else None

        self.horizontal_flip = config["horizontal_flip"] if config.__contains__("horizontal_flip")   else None
        self.vertical_flip = config["vertical_flip"] if config.__contains__("vertical_flip")   else None
        self.rotation_num  = config["rotation_num"]  if config.__contains__("rotation_num")   else None
        self.crop_size     = config["crop_size"]  if config.__contains__("crop_size")   else None
        self.crop_num      = config["crop_num"]  if config.__contains__("crop_num")   else None

        # self.patch_size    = config["patch_size"]  if config.__contains__("patch_size")   else None
        # self.stride        = config["stride"]       if config.__contains__("stride")        else None

        self.normalization = config["normalization"]if config.__contains__("normalization") else None
        
    def get_dataset(self):
        label_np          = self.load_label_np()
        label_patches_np  = self._preprocess_data(label_np)
        train_patches_np  = self.processor.add_gaussian_noise(label_patches_np, self.gauss_sigma/255)
        dataset           = self.generate_loader_from_np(train_patches_np,label_patches_np)
        return dataset

    def _preprocess_data(self,data_np):
        """
        This function may be deprecated in the future
        """
        #
        data_np = self.processor.process_colorspace(data_np,self.colorspace)    if self.colorspace is not None else data_np

        # augmentation
        data_np = self.processor.vertical_flip(data_np)                          if self.vertical_flip is  True else data_np
        data_np = self.processor.horizontal_flip(data_np)                        if self.horizontal_flip is  True else data_np
        data_np = self.processor.rand_rotation(data_np,num=self.rotation_num)    if self.rotation_num is not None else data_np

        data_np = self.processor.random_crop(data_np,self.crop_size,self.crop_num) if self.crop_size is not None else data_np

        data_np = self.processor.normalization_1(data_np)                         if self.normalization is True else data_np
        #data_np = self.processor.get_patches(data_np,self.patch_size,self.stride)    if self.patch_size is not None else data_np
        return data_np

class DenoisingTestData(BaseTestData):
    def __init__(self, config=None):
        super().__init__(config)
        self.gauss_sigma = config["gauss_sigma"] if config.__contains__("gauss_sigma") else 1
        self.colorspace = config["colorspace"] if config.__contains__("colorspace") else None

        self.patch_size    = config["patch_size"]  if config.__contains__("patch_size")   else None
        self.stride        = config["stride"]       if config.__contains__("stride")        else None

        self.normalization = config["normalization"] if config.__contains__("normalization") else None

    def _preprocess_data(self, data_np):
        """
        This function may be deprecated in the future
        """
        #
        data_np = self.processor.process_colorspace(data_np,
                                                    self.colorspace) if self.colorspace is not None else data_np
        data_np = self.processor.normalization_1(data_np)                         if self.normalization is True else data_np

        data_np = self.processor.get_patches(data_np,self.patch_size,self.stride)    if self.patch_size is not None else data_np
        return data_np

    def get_test_data(self,image_name):
        image_path = self.input_path + '\\' + image_name
        image_np = self.dataloader.cvread_to_numpy(image_path, 'gray')
        origin_shape = [image_np.shape[2], image_np.shape[3]]

        label_patches_np = self._preprocess_data(image_np.reshape(1,1,origin_shape[0],origin_shape[1]))
        input_patches_np = self.processor.add_gaussian_noise(label_patches_np, self.gauss_sigma / 255)
        # noise_np         = input_patches_np - label_patches_np
        return input_patches_np,label_patches_np,origin_shape
