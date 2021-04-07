import sys
sys.path.insert(0,"H:\\HCX\\PETreconstruction36\\Projects3\\Algorithms\\PET2\\algorithms")

from DIPRecon.DIPRecon import DIPRecon

def selectAlgorithm(algName):
    model = None
    if algName =="DIPRecon":
        model = DIPRecon
    assert model is not None
    return model



