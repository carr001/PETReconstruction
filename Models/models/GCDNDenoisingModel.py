import torch
import numpy as np
import torch.nn as nn
import torchvision.utils as tv_utils

from networks.GCDNNet import GCDNNet
from models.BaseDenoisingModel import BaseTrainModel,BaseTestModel

from utils.modelUtils import getDate,getTime24

class GCDNDenoisingTrainModel(BaseTrainModel):
    def __init__(self,config):
        BaseTrainModel.__init__(self,config)

    def bind_net(self,**config):
        self.net = GCDNNet(**config)

class GCDMTestModel(BaseTestModel):
    def __init__(self,config):
        BaseTestModel.__init__(self,config)

    def bind_net(self,config):
        self.net = GCDNNet(**config['net_config'])

    def __deconfig_net__(self,net_config):
        """
        deconfig net_fig, make all the parameter a list

        :param net_config:
        :return: a list of paramters with the oder of DnCNNNet.__init__()
        """
        net_args = {}
        net_args["nic"] =  net_config["nic"]
        net_args["nf"] =  net_config["nf"]
        net_args["iters"] =  net_config["iters"]
        net_args["window_size"] =  net_config["window_size"]
        net_args["topK"] =  net_config["topK"]
        net_args["rank"] =  net_config["rank"]
        net_args["circ_rows"] =  net_config["circ_rows"]
        net_args["leak"] =  net_config["leak"]
        return net_args

    def writer_image_test(self, estima_torch, labels_torch, num_per_row=1, normalize=True, scale_each=True):
        if self.writer is not None:
            image_posx = getDate() + '_' + getTime24()
            psnr = self.metric(estima_torch.numpy(), labels_torch.numpy(), 1.)
            estima_tag = image_posx + '_estima_test_PSNR='+f'%.4f'%psnr
            label_tag  = image_posx + '_label_test'
            self.writer_image(estima_torch,estima_tag,None,num_per_row=num_per_row, normalize=normalize,scale_each=scale_each)
            self.writer_image(labels_torch,label_tag, None,num_per_row=num_per_row, normalize=normalize,scale_each=scale_each)

    def get_image_patches(self,image_name):
        assert self.data.test_dnp.__contains__(image_name)
        image_path = self.data.input_path + '\\' + image_name
        image_np = self.data.dataloader.cvread_to_numpy(image_path, 'gray')
        origin_shape = [image_np.shape[2], image_np.shape[3]]
        label_patches_np = self.data.preprocess_data(image_np)
        noise_np = (self.data.processor.gauss_sigma / 255) * np.random.randn(*label_patches_np.shape)
        input_patches_np = label_patches_np + noise_np

        return input_patches_np,label_patches_np,origin_shape

    def get_output(self,input_np):
        """
        Batch to avoid insuffcient gpu memory
        :param input_np:
        :return:
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        batch = 12
        for ith,i in enumerate( range(0,input_np.shape[0],batch)):
            end = i+batch if i+batch < input_np.shape[0] else input_np.shape[0]
            input_batch_torch =  torch.from_numpy(input_np[i:end]).float().to(device)
            output_batch_np= self.net.forward(input_batch_torch).detach().to('cpu').numpy()
            output_np = output_batch_np if ith==0 else np.vstack([output_np,output_batch_np])
        return output_np

    def get_aver_psnr(self,dir=None,patch_inverse=False):
        """
        Test psnr on all image
        :param dir:
        :return: average psnr on an dir or data.test_dnp, a list that contain individual psnr value
        """
        psnr_list = []
        if dir ==None:
            for image_name in self.data.test_dnp.keys():
                input_patches_np, label_patches_np, origin_shape = self.get_image_patches(image_name)
                output_patches_np  = self.get_output(input_patches_np)
                if patch_inverse is True:
                    estima_np    = self.data.get_patches_inverse(output_patches_np,origin_shape)
                    label_np     = self.data.get_patches_inverse(label_patches_np,origin_shape)
                else:
                    estima_np    = output_patches_np
                    label_np     = label_patches_np
                psnr_list.append(self.choose_metric('psnr')(estima_np, label_np, 1.))
        else:
            pass
        self.log_and_print(f'  Test average PSNR = %.4f'%(np.sum(psnr_list)/psnr_list.__len__())+' on directory: '+ self.data.input_path)
        return np.sum(psnr_list)/psnr_list.__len__(),psnr_list

    def get_psnr_on(self,image_name,patch_inverse=True,writer_image=False):
        """
        Test psnr on one image
        :param image_name:
        :param writer_image:
        :param patch_inverse:
        :return:
        """
        input_patches_np, label_patches_np, origin_shape = self.get_image_patches(image_name)
        output_patches_np = self.get_output(input_patches_np)

        if patch_inverse is True:
            estima_np    = self.data.get_patches_inverse(output_patches_np,origin_shape)
            label_np     = self.data.get_patches_inverse(label_patches_np,origin_shape)
            psnr = self.choose_metric('psnr')(estima_np, label_np,1.)

            self.writer_image_test(torch.from_numpy(estima_np).float(), torch.from_numpy(label_np).float()) if writer_image is True else []
        else:
            estima_np    = output_patches_np
            label_np     = label_patches_np
            psnr = self.choose_metric('psnr')(estima_np, label_np,1.)

            self.writer_image_test(torch.from_numpy(estima_np).float(), torch.from_numpy(label_np).float(),num_per_row=int(np.sqrt(estima_np.shape[0]))) if writer_image is True else []
        self.log_and_print(f'  Test PSNR = %.4f'% psnr+' on file: '+ self.data.input_path+'\\'+image_name)
        return psnr

if __name__ == "__main__":
    print(GCDNTrainModel.__mro__)









