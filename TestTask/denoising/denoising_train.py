import os
import json

import sys
sys.path.insert(0,os.path.abspath( os.path.join(os.path.dirname(__file__),'../../Models')))

from data.DenoisingData import DenoisingTrainData
from SelectModel import selectTrainModel

def ModelTraining(json_path):
    with open(json_path) as f:
        config = json.load(f)

    model = selectTrainModel(config['model_config'])
    data  = DenoisingTrainData(config['data_config'])

    model.train(data.get_dataset())
    model.detach_gpu()
    print('This serve as a breakpoint')

if __name__ == '__main__':
    os.chdir('H:\\HCX\\PETreconstruction36\\Projects3\\TestTask\\denoising')
    # json_path = 'config/DnCNNModelTrainCONFIG.json'
    # ModelTraining(json_path)
    #
    # json_path = 'config/NLRNModelTrainCONFIG.json'
    # ModelTraining(json_path)

    json_path = 'config/GCDNModelTrainCONFIG.json'
    ModelTraining(json_path)

    print('This serve as a breakpoint')




