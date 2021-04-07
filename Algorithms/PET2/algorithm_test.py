import os
import sys
import json

sys.path.insert(0,"H:\\HCX\\PETreconstruction36\\Projects3\\Algorithms\\PET2")
from SelectAlgorithm import selectAlgorithm

def getAlgorithm(algName,json_path):
    with open(json_path) as f:
        config = json.load(f)
    return selectAlgorithm(algName)(config)

if __name__=="__main__":
    os.chdir('H:\\HCX\\PETreconstruction36\\Projects3\\Algorithms\\PET2')

    DIPRecon_config_path  = 'config/DIPReconCONFIG.json'
    DIPRecon_model = getAlgorithm("DIPRecon",DIPRecon_config_path)
    output = DIPRecon_model.forward_all()

    print('This serve as a breakpoint')









