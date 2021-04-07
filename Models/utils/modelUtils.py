import os
import time
import math
import re
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import skimage.measure as skm

def saveCkpt(path, net=None, opt=None, sched=None):
    """
    code borrow from https://github.com/nikopj/DGCN.git
    """
    getSD = lambda obj: obj.state_dict() if obj is not None else None
    torch.save({
                'net_state_dict': getSD(net),
                'opt_state_dict': getSD(opt),
                'sched_state_dict': getSD(sched)
                }, path)


def loadCkpt(path, net=None,opt=None,sched=None):
    """
    code borrow from https://github.com/nikopj/DGCN.git
    """
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    def setSD(obj, name):
        if obj is not None and name+"_state_dict" in ckpt:
            print(f"Loading {name} state-dict...")
            obj.load_state_dict(ckpt[name+"_state_dict"])
            return obj
    net = setSD(net, 'net')
    opt   = setSD(opt, 'opt')
    sched = setSD(sched, 'sched')
    return net, opt, sched

def getDate():
    t = time.localtime()
    return str(t.tm_year)+'_'+str(t.tm_mon)+'_'+str(t.tm_mday)

def getTime24():
    t = time.localtime()
    return str(t.tm_hour) + '_' + str(t.tm_min)

def emptyFilesOf(dir):
    if os.path.isdir(dir):  # 如果目录下有文件了，就删除目录下所有文件
        content = os.listdir(dir)
        for ele in content:
            filename = dir +'/'+ ele
            os.remove(filename)

def findSavedPth(dir,epoch):
    """
    指定目录下寻找某次迭代后保存的模型
    :param dir: the directory to look for
    :param epoch: which epoch to return
    :return:
    """
    if os.path.isdir(dir):
        files = os.listdir(dir)
        for file in files:
            if "epoch"+str(epoch) in file:
                return file
    else:
        #os.mkdir(dir) 只能创建单级目录
        os.makedirs(dir)
        return None

def chooseLossFunc(loss_func_name):
    if loss_func_name == 'MSE':
        return nn.MSELoss(size_average=False)
    else:
        print('The loss function you want is not included')
        raise NotImplementedError

def chooseOptimizer(net,optimizer_name,**kargs):
    optimizer =None
    if optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(),**kargs)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(net.parameters(),**kargs)
    else:
        print('The optimizer you choose is not included')
        raise NotImplementedError
    return optimizer

def chooseInitializer(initializer_name):
    if initializer_name == "kaiming":
        return weights_init_kaiming

def chooseSched(optimizer,**kargs):
    pass
    return None

def chooseMetric(metric_str=None):
    if metric_str == 'psnr':
        return psnr
    elif metric_str is None:
        pass
    else:
        print('The Metric you want is not implemented')
        raise NotImplementedError

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)


def multiplyList(list):
    assert list.__len__()>0
    product = 1
    for ele in list:
        product = product*ele
    return product

def psnr(signal,ground_truth,data_range=None):
    """
    Formular reference to https://zhuanlan.zhihu.com/p/50757421
    :param signal:signal with shape [shape[0]] or [shape[0],shape[1]] or [shape[0],shape[1],shape[2]]
    :param ground_truth:
    :return:
    """
    # mse_ = mse(signal,ground_truth)
    # psnr = 20*math.log10(255/mse_)
    return skm.compare_psnr(ground_truth,signal,data_range)

def psnrs(signals,ground_truths,data_range=None):
    """
    Formular reference to https://zhuanlan.zhihu.com/p/50757421
    :param signal:signal with shape [shape[0]] or [shape[0],shape[1]] or [shape[0],shape[1],shape[2]]
    :param ground_truth:
    :return:
    """
    assert len(signals.shape) == 4
    num_of_data = signals.shape[0]
    psnr_sum  = 0
    for i in range(num_of_data):
        psnr_     = psnr(signals[i],ground_truths[i],data_range)
        psnr_sum = psnr_sum + psnr_
    return psnr_sum / num_of_data

def mses(signals,ground_truths):
    """

    :param signals: signal with shape[num_of_data,shape[0],shape[1],shape[2]]
    :param ground_truths:
    :return:
    """
    assert len(signals.shape) > 4
    num_of_data = signals.shape[0]
    mse_sum = 0
    for i in range(num_of_data):
        mse_     = mse(signals[i],ground_truths[i])
        mse_sum = mse_sum + mse_
    return mse_sum / num_of_data

def mse(signal,ground_truth):
    """
    :param signal: signal with shape [shape[0]] or [shape[0],shape[1]] or [shape[0],shape[1],shape[2]]
    :param ground_truth:
    :return:
    """
    sum = np.sum((signal-ground_truth)**2)
    mse = sum/multiplyList(signal.shape)
    return mse



