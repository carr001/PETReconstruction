from models.DnCNNDenoisingModel import DnCNNDenoisingTrainModel,DnCNNDenoisingTestModel
from models.GCDNDenoisingModel import GCDNDenoisingTrainModel
from models.NLRNDenoisingModel import NLRNDenoisingTrainModel
from models.UNetDIPDenoisingModel import UNetDIPDenoisingModel
from models.DnCNNDIPDenoisingModel import DnCNNDIPDenoisingModel
from models.AEDDIPDenoisingModel import AEDDIPDenoisingModel

def selectTrainModel(model_config):
    model = None
    modelName = model_config['model_name']
    if modelName =="DnCNNModel":
        model = DnCNNDenoisingTrainModel(model_config)
    if modelName =="GCDNModel":
        model = GCDNDenoisingTrainModel(model_config)
    if modelName =="NLRNModel":
        model = NLRNDenoisingTrainModel(model_config)
    assert model is not None
    return model

def selectTestModel(model_config):
    model = None
    modelName = model_config['model_name']
    if modelName =="DnCNNModel":
        model = DnCNNDenoisingTestModel(model_config)

    # if modelName =="GCDNModel":
    #     model = GCDNDenoisingTestModel(model_config)
    # if modelName =="NLRNModel":
    #     model = NLRNDenoisingTestModel(model_config)
    if modelName =="UNetDIPModel":
        model = UNetDIPDenoisingModel(model_config)
    if modelName == "DnCNNDIPModel":
        model = DnCNNDIPDenoisingModel(model_config)
    if modelName == "AEDDIPModel":
        model = AEDDIPDenoisingModel(model_config)
    assert model is not None
    return model


