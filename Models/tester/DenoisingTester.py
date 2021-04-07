import torch
import numpy as np

from tqdm import tqdm
from tester.BaseTester import BaseTester
from utils.modelUtils import findSavedPth,getDate,getTime24,chooseLossFunc,chooseOptimizer,chooseMetric,chooseInitializer,chooseSched,saveCkpt,loadCkpt

class DenoisingTester(BaseTester):
    def __init__(self,model,**config):
        super().__init__(**config)
        self.model = model
        self.metric_name = config["metric_name"]  if config.__contains__("metric_name") else None
        self.metric = chooseMetric(self.metric_name)

    def change_metric(self,metric_name):
        if metric_name != self.metric_name:
            self.metric = chooseMetric(self.metric_str)
        self.model._logger.print("Using metric: "+ metric_name)

    def get_output(self,input_np):
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.model.net.to(device)
        BATCH = input_np.shape[0]
        sub_batch = BATCH
        for ith,start in enumerate(tqdm(range(0,input_np.shape[0],sub_batch),"Geting output...") ):
            end = start+sub_batch if start+sub_batch < input_np.shape[0] else input_np.shape[0]
            input_batch_torch =  torch.from_numpy(input_np[start:end]).float().to(device)
            output_batch_np   = self.model.net.forward(input_batch_torch).detach().to('cpu').numpy()
            output_np = output_batch_np if ith==0 else np.vstack([output_np,output_batch_np])
        return output_np
    def get_aver_metric(self,dir=None,whole_image=False):
        """
        Test psnr on all image
        :param dir:
        :return: average psnr on an dir or data.test_dnp, a list that contain individual psnr value
        """
        psnr_list = []
        if dir == None:
            for image_name in self.model.data.test_dnp.keys():
                psnr,_  = self.get_metric_on(image_name,whole_image)
                psnr_list.append(psnr)
        else:
            pass
        self.model._logger.print(f'  Test average PSNR = %.4f'%(np.sum(psnr_list)/psnr_list.__len__())+' on directory: '+ self.model.data.input_path)
        return np.sum(psnr_list)/psnr_list.__len__(),psnr_list

    def get_metric_on(self,image_name,whole_image=False):
        """
        Test psnr on one image
        :param image_name:
        :param writer_image:
        :param patch_inverse:
        :return:
        """
        input_patches_np, label_patches_np, origin_shape = self.model.data.get_test_data(image_name)
        output_patches_np = self.get_output(input_patches_np)

        if whole_image is True:
            estima_np    = self.model.data.processor.get_patches_inverse(output_patches_np,origin_shape,self.model.data.patch_size,self.model.data.stride)
            label_np     = self.model.data.processor.get_patches_inverse(label_patches_np,origin_shape,self.model.data.patch_size,self.model.data.stride)
            input_np     = self.model.data.processor.get_patches_inverse(input_patches_np,origin_shape,self.model.data.patch_size,self.model.data.stride)
        else:
            estima_np    = output_patches_np
            label_np     = label_patches_np
            input_np     = input_patches_np
        psnr = self.metric(estima_np, label_np,1.)
        return psnr,estima_np,label_np,input_np

    def evaluate_and_log(self,image_name,whole_image=False):
        psnr, estima_np, label_np,input_np = self.get_metric_on(image_name, whole_image)
        self.model._logger.writer.writer_grid_image(torch.from_numpy(estima_np).float(),f'output_PSNR=%.4f'%(psnr),
                               num_per_row=int(np.sqrt(estima_np.shape[0])))

        self.model._logger.writer.writer_grid_image(torch.from_numpy(label_np).float(),'label',
                                       num_per_row=int(np.sqrt(estima_np.shape[0])))

        self.model._logger.writer.writer_grid_image(torch.from_numpy(input_np).float(), 'noisy',
                                                    num_per_row=int(np.sqrt(estima_np.shape[0])))

class DenoisingDIPTester(BaseTester):
    def __init__(self,model,**config):
        super().__init__(**config)
        self.model = model
        self.metric_name = config["metric_name"]  if config.__contains__("metric_name") else None
        self.metric = chooseMetric(self.metric_name)

    def denoise(self,image_name):
        """
        Input is defalut to be noise
        :param image_name:
        :return:
        """
        labels_np, clean_np, _ = self.model.data.get_test_data(image_name)
        output                 = self.model.trainer.train_DIP(self.model,labels_np)
        return output,labels_np, clean_np

    # def denoise2(self,label_np,end_epoch=None,input=None):
    #     output                 = self.model.trainer.train_DIP(self.model,input,label_np,end_epoch)

    def evaluate_and_log(self,image_name):
        estima_np, input_np, labels_np = self.denoise(image_name)
        psnr                           = self.metric(estima_np, labels_np, 1.)

        self.model._logger.writer.writer_grid_image(torch.from_numpy(estima_np).float(),f'output_PSNR=%.4f'%(psnr),
                               num_per_row=int(np.sqrt(estima_np.shape[0])))

        self.model._logger.writer.writer_grid_image(torch.from_numpy(labels_np).float(),'label',
                                       num_per_row=int(np.sqrt(estima_np.shape[0])))

        self.model._logger.writer.writer_grid_image(torch.from_numpy(input_np).float(), 'noisy',
                                                    num_per_row=int(np.sqrt(estima_np.shape[0])))
        pass

