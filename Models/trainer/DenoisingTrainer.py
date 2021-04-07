import torch
import numpy as np
from tqdm import tqdm
from trainer.BaseTrainer import BaseTrainer

class DenoisingTrainer(BaseTrainer):
    def __init__(self,**config):
        super().__init__(**config)

    def train(self,model,dataset):
        assert model.pretrain_epoch <= model.end_epoch
        model._logger.print(f"Start training from epoch { model.pretrain_epoch}")
        model._logger.print(f"Target training epoch { model.end_epoch}")
        for epoch in range(1 + model.pretrain_epoch, model.end_epoch + 1):
            running_loss = 0.0
            model.curr_epoch = epoch
            for traind, label in tqdm(dataset):
                traind, label = traind.to(model.curr_device()), label.to(model.curr_device())

                model.optimizer.zero_grad()
                output = model.net.forward(traind)
                loss = model.loss_func(output, label)
                loss.backward()
                model.optimizer.step()
                model.sched.step() if model.sched is not None else None

                running_loss += loss.item()

                model._logger.writer_scaler(output.detach().to('cpu').numpy(), label.detach().to('cpu').numpy(), loss.item(),curr_epoch=epoch)
            self.save_model(model)

    def train_DIP(self,model,label_np,traind_np=None,end_epoch=None,mask=None):
        model.end_epoch = end_epoch if end_epoch is not None else model.end_epoch
        mask            = mask if end_epoch is not None else np.zeros_like(label_np,dtype=np.bool)
        traind_np       = model.data.processor.add_gaussian_noise(np.zeros_like(label_np), self.model.data.gauss_sigma / 255) if traind_np is None else traind_np

        assert model.pretrain_epoch <= model.end_epoch
        model._logger.print(f"Start training from epoch { model.pretrain_epoch}")
        model._logger.print(f"Target training epoch { model.end_epoch}")

        traind = torch.from_numpy(traind_np).float().to(model.curr_device())
        label  = torch.from_numpy(label_np).float().to(model.curr_device())
        mask   = torch.from_numpy(mask).to(model.curr_device())
        for epoch in range(1 + model.pretrain_epoch, model.end_epoch + 1):
            running_loss = 0.0
            model.curr_epoch = epoch

            model.optimizer.zero_grad()
            output = model.net.forward(traind)
            output = output.masked_fill(mask=mask, value=0)
            loss   = model.loss_func(output, label)
            loss.backward()
            model.optimizer.step()
            model.sched.step() if model.sched is not None else None

            running_loss += loss.item()

            model._logger.writer_scaler(output.detach().to('cpu').numpy(), label.detach().to('cpu').numpy(),
                                        loss.item(), curr_epoch=epoch)
        return output.detach().to('cpu').numpy()