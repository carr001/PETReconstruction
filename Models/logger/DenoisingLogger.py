
from logger.BaseLogging import BaseLogging
from logger.BaseWriter import BaseWriter
from utils.modelUtils import chooseMetric

class DenoisingLogger():
    """
    DenoisingLogger is a wraper of logger and writer
    """
    def __init__(self,log_frequency,metric_name="psnr",logging_config=None,writer_config=None,**kargs):
        self.logging        = BaseLogging(**logging_config)
        self.writer        = BaseWriter(**writer_config)
        self.step_count    = 0
        self.log_frequency = log_frequency
        self.metric_name    = metric_name
        self.metric        = chooseMetric( metric_name)

    def __step(self):
        self.step_count += 1

    def __time_up(self):
        return True if self.step_count % self.log_frequency == 0 else False

    def print(self,args):
        self.logging.print(args)

    def writer_scaler(self, estima_np=None, labels_np=None, running_loss=None,curr_epoch=None):
        """
        This is written for a specific trainer
        """
        self.__step()
        if estima_np is not None and self.__time_up():
            loss = self.metric(estima_np, labels_np, 1.)
            self.writer.add_scalar(self.metric_name, loss, self.step_count)
            self.writer.add_scalar('Running Loss', running_loss, self.step_count)
            self.print(' net epoch: [%d], loss: %.4f, psnr: %.4f' % (curr_epoch, running_loss, loss))

    def writer_image(self, estima_torch, labels_torch,curr_epoch=None, num_per_row=1, normalize=True, scale_each=True):
        """
        This is written for a specific trainer
        """
        self.__step()
        if self.__time_up():
            loss = self.metric(estima_torch.numpy(), estima_torch.numpy(), 1.)
            estima_tag = 'estima_epoch'+ str(curr_epoch)+'_',self.metric_name,f'= %.4f'%loss
            label_tag  = 'label_epoch' + str(curr_epoch)
            self.writer.writer_grid_image(estima_torch,estima_tag,curr_epoch,num_per_row=num_per_row, normalize=normalize,scale_each=scale_each)
            self.writer.writer_grid_image(labels_torch,label_tag, curr_epoch,num_per_row=num_per_row, normalize=normalize,scale_each=scale_each)

    def add_image(self,image, global_step=None,tags=None):
        self.__step()
        if self.__time_up():
            self.writer.add_image(tags,image, global_step)


