import numpy as np
import json
import copy

from utils import calculate_weight, calculate_norm, calculate_s_score_new, calculate_sparsity, calculate_vg_score_new, \
    check_dic_key, write_log
from grow.growth_utils import find_modules


class AnalysisRecorder:
    def __init__(self, optimizer, optimizer_mode, recorder, path):
        self.recorder_dic = {}
        self.optimizer = optimizer
        self.optimizer_mode = optimizer_mode
        self.recorder = recorder
        self.batch_loss = []
        self.test_loss = []
        self.test_accuracy = []
        self.time_in_epoch = []
        self.train_loss = []
        self.time_in_epoch = []
        self.batch_accuracy = []
        self.txt_path = path

    def get_recorder_dic(self):
        return self.recorder_dic

    def get_train_loss(self):
        return self.train_loss

    def record_test_acc(self, acc):
        self.test_accuracy.append(acc)

    def record_batch_accuracy(self, acc):
        self.batch_accuracy.append(acc)

    def record_test_loss(self, loss):
        self.test_loss.append(loss)

    def record_batch_loss(self, epoch):
        str = '\nEpoch_{} = {};\n'.format(epoch, self.train_loss)
        write_log(str, self.txt_path)
        self.train_loss = []

    def record_train_loss(self, loss):
        self.train_loss.append(loss)

    def json_dump(self, path):
        with open(path, 'a') as f:
            json.dump(self.recorder_dic, f, indent=4)
        keys = []
        for key in self.recorder_dic.keys():
            keys.append(key)

        for key in keys:
            del self.recorder_dic[key]

    def record(self, model, epoch, batch):
        dic_exist = {'out', 'layer', 'in'}
        check_dic_key(self.recorder_dic, epoch)
        check_dic_key(self.recorder_dic[epoch], batch)

        layers = find_modules(model)
        for i, layer in enumerate(layers):
            # self.recorder_dic[epoch][batch][i] = {'W': {}, 'L1': {}, 'L2': {}, 'S': {}, 'VG': {}, 'ACV': {},
            #                                       'Sparsity': {}, 'Delta_S': [], 'Cosine': [], 'V': 0,
            #                                       'M': 0, 'G': 0, 'Grad': 0}
            self.recorder_dic[epoch][batch][i] = {'S': {}}

            # V, M, G score
            # if self.optimizer is not None and self.optimizer_mode == 'AdamW' and self.optimizer.vs:
            #     self.recorder_dic[epoch][batch][i]['V'] = self.optimizer.vs[i]
            #     self.recorder_dic[epoch][batch][i]['M'] = self.optimizer.ms[i]
            #     self.recorder_dic[epoch][batch][i]['G'] = self.optimizer.gs[i]
            #     self.recorder_dic[epoch][batch][i]['Grad'] = self.optimizer.grads[i]

            # Delta S
            # if self.recorder.get_pre_score():
            #     if i < len(layers) - 1:
            #         pre = self.recorder.get_pre_score()[i]
            #         cur = self.recorder.get_score()[i]
            #         if len(cur) == len(pre):
            #             delta_s = np.asarray(cur) - np.asarray(pre)
            #             self.recorder_dic[epoch][batch][i]['Delta_S'] = str(delta_s.tolist())

            for key in dic_exist:
                # l1, s = calculate_weight(layer, mode=key)
                s_result = calculate_s_score_new(layer, mode=key)
                # l1_result = calculate_norm(layer, mode=key, order=1)
                # l2_result = calculate_norm(layer, mode=key, order=2)
                # sparsity = calculate_sparsity(layer, mode=key)
                # self.recorder_dic[epoch][batch][i]['W'][key] = {}
                # self.recorder_dic[epoch][batch][i]['W'][key]['L1'] = str(l1)
                # self.recorder_dic[epoch][batch][i]['W'][key]['std'] = str(s)
                # self.recorder_dic[epoch][batch][i]['L1'][key] = str(l1_result)
                # self.recorder_dic[epoch][batch][i]['L2'][key] = str(l2_result)
                self.recorder_dic[epoch][batch][i]['S'][key] = str(s_result)
                # self.recorder_dic[epoch][batch][i]['Sparsity'][key] = str(sparsity)

                # if self.optimizer_mode == 'AdamW' and self.optimizer.vg:
                #     vg_result = calculate_vg_score_new(layer, self.optimizer.vg[i], mode=key)
                #     self.recorder_dic[epoch][batch][i]['VG'][key] = str(vg_result)
                # else:
                #     self.recorder_dic[epoch][batch][i]['VG'][key] = []

    def write_txt(self):
        # write_log(time_str, dic_path['path_to_log'])
        # write_log(para_str, dic_path['path_to_log'])
        # write_log('batch time:{}\n'.format(batch_time_list), dic_path['path_to_log'])
        # write_log('batch test acc:{}\n'.format(batch_test_acc), dic_path['path_to_log'])
        write_log('train time in each epoch:{}\n'.format(self.time_in_epoch), self.txt_path)
        write_log('Loss:{}\n'.format(self.test_loss), self.txt_path)
        write_log('Accuracy:{}\n'.format(self.test_accuracy), self.txt_path)
        write_log('Batch_accuracy:{}\n'.format(self.batch_accuracy), self.txt_path)
        for index, item in enumerate(self.batch_loss):
            str = '\nEpoch_{} = {};\n'.format(index + 1, item)
            write_log(str, self.txt_path)
