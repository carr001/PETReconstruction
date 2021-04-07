import os
import sys
sys.path.insert(0,os.path.abspath( os.path.join(os.path.dirname(__file__),'../../../../Models')))
import numpy as np
import scipy.io as sio

from tqdm import tqdm
from solver.ADMMSolver import ADMMSolver
from data.DataProcessor import DataProcessor
from SelectModel import selectTestModel

print("Starting matlab engine ...")
os.chdir(os.path.dirname(__file__))
import matlab
import matlab.engine as matlab_engine
mlb = matlab_engine.start_matlab()
print("Start matlab engine Done !")


class DIPRecon(ADMMSolver):
    """"""
    def __init__(self,config):
        self.totaliters = config["totaliters"]
        self.subiter1 = config["subiter1"]
        self.subiter2 = config["subiter2"]
        self.pretrain_epoch = config["pretrain_epoch"]
        self.pretrain_iters = config["pretrain_iters"]
        self.frame    = config["frame"]
        self.net_input_config= config["net_input"]
        self.image_shape= config["image_shape"]

        self.model  = None
        self.model_cofig = config["model_config"]

        # for multi realizations
        self.realizations= config["realizations"]
        self.curr_real= 0
        self.z_k_list_allreals = None

        self.rho = config["rho"]
        self.x_0 = self.mlem(iters=10)
        # self.x_0 = []
        self.z_0 = []
        self.w_0 = 0
        super().__init__(self.x_0,self.z_0,self.w_0,self.rho,self.totaliters)

    def initial_net(self,config):
        # self.model = UNetDIPDenoisingModel(config)
        self.model = selectTestModel(config)

    def mlem(self,x_k_1=[],z_k_1=[],w_k_1=[],rho=0,iters=1):
        x_k_1 = x_k_1 if x_k_1 == [] else np.array(x_k_1).reshape(-1,1).tolist()
        z_k_1 = z_k_1 if z_k_1 == [] else np.array(z_k_1).reshape(-1,1).tolist()
        w_k_1 = w_k_1 if w_k_1 == [] else np.array(w_k_1).reshape(-1,1).tolist()

        x_k = mlb.mlem_regularized(matlab.double(x_k_1),matlab.double(z_k_1),matlab.double(w_k_1),matlab.double([rho]),matlab.int8([iters]),matlab.int8([self.frame]),matlab.int8([self.curr_real+1]))
        return np.array( x_k).reshape(1,1,self.image_shape[0],self.image_shape[1])

    def net_train(self,label,end_epoch):
        label,max,min = DataProcessor().normalization_range(label)
        output = self.model.trainer.train_DIP(self.model, label, self.net_input, end_epoch,self.mask)
        return DataProcessor().normalization_range_reverse(output,max,min)

    def initialization_step(self):
        self.curriter=0
        ## data initialization
        if self.net_input_config == "mr":
            os.chdir(os.path.dirname(__file__))
            mr = sio.loadmat("./mr.mat")["mr"]
            self.net_input,_,_ = DataProcessor().normalization_range(mr.reshape(1,1,self.image_shape[1],self.image_shape[0]).transpose(0,1,3,2))# the shape order has been checked!
        else:
            self.net_input = np.random.randn(1,1,self.image_shape[0],self.image_shape[1])
        os.chdir(os.path.dirname(__file__))
        self.mask = ~sio.loadmat("./u.mat")["u"].reshape(1,1,self.image_shape[1],self.image_shape[0]).transpose(0,1,3,2).astype(np.bool)

        ## net initialization
        self.initial_net(self.model_cofig)
        # x_pre = self.mlem(iters= self.pretrain_iters)
        self.net_train(self.x_0,self.pretrain_epoch)

    def forward(self):
        self.initialization_step()
        output = self.solve()
        self.save_iter_result()
        return output

    def forward_all(self):
        for self.curr_real in tqdm(range(self.realizations),'Runing multiple realizations'):
            self.initialization_step()
            self.solve()
            self.model.detach_gpu()
            self.save_one_real()
        self.save_all_reals()

    ################# realization parent's virtual method #######################
    def sovle_x(self,x_k_1,z_k_1,w_k_1):
        return self.mlem(x_k_1,z_k_1,w_k_1,rho=self.rho,iters=self.subiter1)

    def solve_z(self,x_k,z_k_1,w_k_1):
        z_k = self.net_train( x_k+w_k_1,self.subiter2)
        return z_k

    ########################### save and log ##############################

    def save(self):
        if self.save_iters is True:
            self.z_k_list = self.z_k if self.curriter ==0 else np.vstack([self.z_k_list,self.z_k])

    def save_one_real(self):
        self.z_k_list = self.z_k_list.reshape(-1, self.image_shape[0] * self.image_shape[1]).transpose()[np.newaxis, :]
        self.z_k_list_allreals = self.z_k_list if self.curr_real == 0 else np.vstack(
            [self.z_k_list_allreals, self.z_k_list])
        self.z_k_list = None

    def save_iter_result(self):
        if self.save_iters is True:
              sio.savemat("diprecon.mat",{"diprecon":self.z_k_list.reshape(-1,self.image_shape[0]*self.image_shape[1]).transpose()})

    def save_all_reals(self):
        sio.savemat("diprecons.mat",
                    {"diprecons": self.z_k_list_allreals})
    def log(self):
        self.model._logger.writer.add_image(self.z_k[0],tags=str(self.curriter))
