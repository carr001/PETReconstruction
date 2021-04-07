from networks.DnCNNNet import DnCNNNet
from models.BaseDenoisingModel import BaseTrainModel,BaseTestModel

class DnCNNDenoisingTrainModel(BaseTrainModel):
    def __init__(self,config):
        BaseTrainModel.__init__(self,config)

    def bind_net(self,**config):
        self.net = DnCNNNet(**config)

class DnCNNDenoisingTestModel(BaseTestModel):
    def __init__(self,config):
        BaseTestModel.__init__(self,config)

    def bind_net(self,**config):
        self.net = DnCNNNet(**config)


if __name__ == "__main__":
    print(DnCNNTrainModel.__mro__)









