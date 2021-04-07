from networks.DnCNNNet import DnCNNNet
from models.BaseDenoisingModel import BaseTrainModel,BaseTestModel
from tester.DenoisingTester import DenoisingDIPTester

class DnCNNDIPDenoisingModel(BaseTrainModel,BaseTestModel):
    def __init__(self,config):
        super().__init__(config)
        #BaseTestModel.__init__(self,config)

    def bind_net(self,**config):
        self.net = DnCNNNet(**config)

    def bind_tester(self,**config):
        self.tester = DenoisingDIPTester(self,**config)

if __name__ == "__main__":
    print(DnCNNTrainModel.__mro__)









