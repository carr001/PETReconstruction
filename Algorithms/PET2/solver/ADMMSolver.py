import numpy as np
from tqdm import tqdm

class ADMMSolver():
    """
    Only support one constrain currently
    """
    def __init__(self,x_0,z_0,w_0,rho,totaliters,save_iters = True,**kargs):
        self.x_0 = x_0
        self.z_0 = z_0
        self.w_0 = w_0
        self.rho = rho
        self.curriter = 0
        self.totaliters = totaliters
        self.save_iters = save_iters
        self.x_k_list = None
        self.z_k_list = None
        self.w_k_list = None

    def sovle_x(self,x_k_1,z_k_1,w_k_1):
        raise NotImplementedError
    def solve_z(self,x_k,z_k_1,w_k_1):
        raise NotImplementedError
    def solve_w(self,x_k,z_k,w_k_1):
        return w_k_1+x_k-z_k
    def log(self):
        raise NotImplementedError
    def save(self):
        raise NotImplementedError


    def solve(self):
        self.x_k =  self.sovle_x(self.x_0,self.z_0,self.w_0)
        self.z_k =  self.solve_z(self.x_k,self.z_0,self.w_0)
        self.w_k =  self.solve_w(self.x_k,self.z_k,self.w_0)
        self.save()
        self.log()
        for self.curriter in tqdm(range(1,self.totaliters),'Runing ADMM'):
            self.x_k = self.sovle_x(self.x_k, self.z_k, self.w_k)
            self.z_k = self.solve_z(self.x_k, self.z_k, self.w_k)
            self.w_k = self.solve_w(self.x_k, self.z_k, self.w_k)
            self.save()
            self.log()
        return self.x_k,self.z_k,self.w_k



