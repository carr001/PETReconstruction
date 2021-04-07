
class BaseTrainer():

    def __init__(self, save_frequency=10):
        self.epoch_count    = 0
        self.save_frequency = save_frequency

    def __time_up(self):
        return True if self.epoch_count % self.save_frequency == 0 else False

    def save_model(self,model):
        self.epoch_count += 1
        if self.__time_up():
            model.save_this_model()

    def train(self):
        print('train not implemented')
        raise NotImplementedError




