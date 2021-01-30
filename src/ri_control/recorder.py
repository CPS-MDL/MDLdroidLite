from utils import calculate_s_score_new, calculate_weight, check_dic_key
from grow.growth_utils import find_modules_short


class Recorder:
    def __init__(self, size=0):

        self.score = []
        self.pre_score = []
        self.weight = []
        self.pre_weight = []
        self.num_layers = size
        self.epoch_score = {}
        self.init_epoch_score()

    def record(self, model):
        s = []
        w = []
        modules = find_modules_short(model)
        for i, module in enumerate(modules):
            new_score = calculate_s_score_new(module, mode='out')
            s.append(new_score)

            new_weight, _ = calculate_weight(module, mode='out')
            w.append(new_weight)

            # layer score
            layer_score = calculate_s_score_new(module, mode='layer')
            self.epoch_score[i].append(layer_score)

        if self.score:
            self.pre_score = self.score

        if self.weight:
            self.pre_weight = self.weight

        self.score = s
        self.weight = w

    def get_pre_score(self):
        return self.pre_score

    def get_score(self):
        return self.score

    def get_weight(self):
        return self.weight

    def get_pre_weight(self):
        return self.pre_weight

    def get_epoch_score(self):
        return self.epoch_score

    def init_epoch_score(self):
        for i in range(self.num_layers):
            self.epoch_score[i] = []


