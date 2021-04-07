import os
import cv2
from tqdm import tqdm
import numpy as np


class Dataloader(object):
    """
    Mainly in charge of load data from various types, such dir, h5 etc
    Dataloader currently support load data from
    1.a directory that contain images

    """
    def __init__(self):
        super().__init__()

    def choose_imread_flag(self,colorspace=None):
        return cv2.IMREAD_UNCHANGED if colorspace is None else cv2.IMREAD_GRAYSCALE if colorspace == 'gray' else cv2.IMREAD_COLOR

    def cvread_to_numpy(self,image_path,colorspace=None):
        """
        :param image_path: path of an image
        :param colorspace:
        :return: numpy array with shape [1,1,shape[0],shape[1]] or [1,3,shape[0],shape[1]]
        """
        if colorspace is None:
            image = np.array(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
        elif colorspace == 'gray':
            image = np.array(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        else:
            image = np.array(cv2.imread(image_path, cv2.IMREAD_COLOR))
        if len(image.shape) == 2:
            return image[np.newaxis,np.newaxis,:,:]
        else:
            return image.transpose([2,0,1])[np.newaxis, :]
    def load_images_from_dir(self,path,colorspace=None):
        """
        This function assume file in path are all images with the same size
        :param path: dir that contain images with the same size
        :return: np array with shape [num_of_images, channels, shape0, shape1]
        """
        image_names = os.listdir(path)
        input_np = []
        for i,image_name in  enumerate(tqdm(image_names,"Loading")):
            image_path = path + "\\" +image_name
            image_np = self.cvread_to_numpy(image_path,colorspace)
            if i == 0:
                input_np = image_np
            else:
                input_np =np.vstack([input_np, image_np])
        return  input_np

    def load_images_to_dict(self,path,colorspace=None):
        """
        Load files in an dir in to a dictionary, keys are corresponding file name
        :param path:
        :return:
        """
        image_names = os.listdir(path)
        dict = {}
        for image_name in tqdm(image_names,"Loading") :
            image_path = path + "\\" +image_name
            dict[image_name] = self.cvread_to_numpy(image_path,colorspace)
        return dict
