import cv2
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import Image

class DataProcessor(object):
    """
    This class is served to process data
    before any precess, a preprocess is apply according to the configuration
    """
    def __init__(self):
        super().__init__()

    def get_patches_inverse(self,images_patches_np,origin_shape,patch_size=None,stride=None):
        """
        The inverse operation of get_patches,stride must equal 1 because only in this way we can reconstruct the image
        :param images_patches_np: numpy array with shape [num_of_patches,channels,shape[0],shape[1]]
        :param origin_shape: list with shape [image_shape[0],image_shape[1]]
        :param patch_size:
        :param stride: must equal patch_size
        :return:
        """
        assert origin_shape is not None

        if patch_size is not None:
            assert images_patches_np.shape[2] == patch_size and images_patches_np.shape[3] == patch_size and stride == patch_size
            patches_per_row = range(0,origin_shape[0]-patch_size+1,stride).__len__()
            patches_per_col = range(0,origin_shape[1]-patch_size+1,stride).__len__()
            image_patches = patches_per_row*patches_per_col
            num_of_data =  int(images_patches_np.shape[0]/image_patches)

            images_np = np.zeros([num_of_data,images_patches_np.shape[1],origin_shape[0],origin_shape[1]])
            # start from left up corrner of an image
            k = 0
            for i in range(0,images_np.shape[2]-patch_size+1,stride):
                for j in range(0,images_np.shape[3]-patch_size+1,stride):
                    images_np[:,:,i:i+patch_size,j:j+patch_size] = images_patches_np[k:k+num_of_data,:,:,:]
                    k = k + num_of_data
            return images_np

    def get_patches(self,images_np,patch_size=None,stride=None):
        """
        Given a numpy dataset, return patches with patch_size*patch_size with stride from CONFIG file or parameters.

        the reason we minus patch_size to compute patches_per_row is because we compute patches from left top corner
        :param images_np: numpy array with shape [num_of_data,channels,shape[0],shape[1]]
        :param patch_size:
        :param stride:
        :return: numpy array with shape [total_patchs,channels,patch_size,patch_size]
        """
        if patch_size is not None and patch_size is not 0:
            assert images_np.shape[2] > patch_size and images_np.shape[3] > patch_size
            num_of_data =  images_np.shape[0]
            patches_per_row = range(0,images_np.shape[2]-patch_size+1,stride).__len__()
            patches_per_col = range(0,images_np.shape[3]-patch_size+1,stride).__len__()
            image_patches = patches_per_row*patches_per_col
            images_patches_np = np.zeros([num_of_data*image_patches,images_np.shape[1],patch_size,patch_size])
            # start from left up corrner of an image
            k = 0
            for i in range(0,images_np.shape[2]-patch_size+1,stride):
                for j in range(0,images_np.shape[3]-patch_size+1,stride):
                    images_patches_np[k:k+num_of_data,:,:,:] = images_np[:,:,i:i+patch_size,j:j+patch_size]
                    k = k+num_of_data
            return images_patches_np
        else:
            return images_np
    def change_colorspace(self,image,colorspace):
        """
        Change the color space of image
        :param image: numpy array with shape [channels,shape[0],shape[1]] or [shape[0],shape[1]] or [1, shape[0],shape[1]]
        :return:  [1,shape[0],shape[1]]  or [channels,shape[0],shape[1]]
        """
        if colorspace == "gray":
            image_ = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)[np.newaxis, :, :]
        else:
            print("Such colorspace transform is not implemented")
            raise NotImplementedError
        return image_

    def process_colorspace(self,images_np,colorspace=None):
        """
        Preprocess colorspace of images_np
        :param images_np:
        :param colorspace:
        :return: processed data
        """
        num_of_data = images_np.shape[0]
        channels = images_np.shape[1]

        images_np_ = images_np
        if colorspace is not None:
            if colorspace == "gray" and channels is not 1:
                channels_ = 1
                images_np_ = np.zeros([num_of_data, channels_, images_np.shape[2], images_np.shape[3]])
                for i in range(num_of_data):
                    images_np_[i] = self.change_colorspace(images_np[i], colorspace)

        return images_np_

    def add_gaussian_noise(self, data_np, sigma=None):
        """
        Add gaussian noise with sigma
        :param data_np: numpy array with shape [num_of_images, channels,shape[0],shape[1]]
        :param sigma: noise level
        :return: noise data
        """
        num_of_data = data_np.shape[0]

        subshape    = [data_np.shape[1],data_np.shape[2],data_np.shape[3]]
        data_np_    = data_np.copy()
        for i in tqdm(range(num_of_data),"Adding gauss noise"):
            data_np_[i]   = data_np[i] + sigma*np.random.randn(*subshape)
        return data_np_

    def add_gaussian_noise_dict(self, data_dnp, sigma=None):
        """
        Add gaussian noise with sigma
        :param data_np: dictionary with data
        :param sigma: noise level
        :return: noise data
        """

        data_dnp_ = data_dnp.copy()
        for key in list(data_dnp.keys()):
            data_dnp_[key]  = data_dnp[key] + sigma * np.random.randn(*data_dnp[key].shape)
        return data_dnp_

    def normalization_range(self,data_np):
        """
        Normalization with method 1
        reference https://blog.csdn.net/chenpe32cp/article/details/81779463
        :param data_np:
        :return:
        """
        max = data_np.max()
        min = data_np.min()
        return ((data_np - min)/(max-min)),max,min

    def normalization_range_reverse(self,data_np,max,min):
        return data_np*(max-min)+min

    def normalization_1(self,data_np):
        """
        Normalization with method 1
        reference https://blog.csdn.net/chenpe32cp/article/details/81779463
        Normalize data to [0,max]
        :param data_np:
        :return:
        """
        count = 0
        data_np_ = np.zeros_like(data_np,dtype=np.float)
        for i,data in  enumerate(tqdm(data_np,"Normalizing")):
            try:
                if data.max() - data.min()==0:
                    raise ZeroDivisionError
                else:
                    data_np_[i] = ((data - data.min())/(data.max() - data.min()))
            except ZeroDivisionError:
                count += 1
                data_np_[i] = np.zeros_like(data)
        print(f'There are %d zero denominators in all %d samples'%(count,data_np.shape[0]))
        return data_np_

    def random_crop(self,data_np,size,num=1):
        """

        :param data_np: […, H, W] shape, where … means an arbitrary number of leading dimensions
        :param size:
        :param num: how many time to crop each sample
        :return:
        """
        data_ = []
        for data in tqdm(data_np,"Random crop...") :
            for _ in range(num):
                data_.append(np.array(transforms.RandomCrop(size)(Image.fromarray(data.squeeze()))))
        return np.array(data_)[:,np.newaxis,:,:] if len(data.squeeze().shape) == 2 else np.array(data_)

    def rand_rotation(self,data_np,degrees=[-180,180],num=1):
        data_ = []
        for data in tqdm(data_np,"Random rotation...") :
            for _ in range(num):
                data_.append(np.array(transforms.RandomRotation(degrees)(Image.fromarray(data.squeeze()))))
        data_ = np.array(data_)[:, np.newaxis, :, :] if len(data.squeeze().shape) == 2 else np.array(data_)
        return np.vstack([data_,data_np])

    def vertical_flip(self,data_np):
        data_ = []
        for data in tqdm(data_np,"Vertical flip...") :
            data_.append(np.array(transforms.RandomVerticalFlip(1)(Image.fromarray(data.squeeze()))))
        data_ = np.array(data_)[:, np.newaxis, :, :] if len(data.squeeze().shape) == 2 else np.array(data_)
        return np.vstack([data_,data_np])
    def horizontal_flip(self,data_np):
        data_ = []
        for data in tqdm(data_np,"Horizontal flip...") :
            data_.append(np.array(transforms.RandomHorizontalFlip(1)(Image.fromarray(data.squeeze()))))
        data_ = np.array(data_)[:, np.newaxis, :, :] if len(data.squeeze().shape) == 2 else np.array(data_)
        return np.vstack([data_,data_np])
