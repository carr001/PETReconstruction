import os
import json

import sys
sys.path.insert(0,os.path.abspath( os.path.join(os.path.dirname(__file__),'../../Models')))

from SelectModel import selectTestModel
from data.DenoisingData import DenoisingTestData

def getConfigedTestModel(json_path):
    with open(json_path) as f:
        config = json.load(f)

    model = selectTestModel(config['model_config'])

    data = DenoisingTestData(config['data_config'])
    model.bind_data(data)
    return model

if __name__ == '__main__':
    os.chdir('H:\\HCX\\PETreconstruction36\\Projects3\\TestTask\\denoising')
    # dncnn_config_path  = 'config/DnCNNModelTestCONFIG.json'
    # dncnn_model = getConfigedTestModel(dncnn_config_path)
    # dncnn_psnr  = dncnn_model.tester.evaluate_and_log('01.png',False)
    # dncnn_psnr,dncnn_psnr_list  = dncnn_model.get_aver_psnr()

    # gcdn_config_path  = 'config/GCDNModelTestCONFIG.json'
    # gcdn_model = getConfigedTestModel(gcdn_config_path)
    # # gcdn_psnr  = gcdn_model.get_psnr_on('01.png', True, True)
    # gcdn_psnr, gcdn_psnr_list   = gcdn_model.get_aver_psnr()
    #
    # nlrn_config_path  = 'config/NLRNModelTestCONFIG.json'
    # nlrn_model = getConfigedTestModel(nlrn_config_path)
    # # nlrn_psnr  = nlrn_model.get_psnr_on('01.png', True, True)
    # nlrn_psnr, nlrn_psnr_list   = nlrn_model.get_aver_psnr()

    unet_config_path  = 'config/UNetDIPModelCONFIG.json'
    unet_model = getConfigedTestModel(unet_config_path)
    unet_psnr  = unet_model.tester.evaluate_and_log('01.png')
    # dncnn_psnr,dncnn_psnr_list  = dncnn_model.get_aver_psnr()
    print('This serve as a breakpoint')




