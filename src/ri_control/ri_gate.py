import torch
import numpy as np
import json
from collections import deque

from utils import check_dic_key
from grow.activation_hook import return_preacv_dic, retrun_acv_dic
from grow.growth_utils import find_modules_short
from utils import device

#############################


def _ri_compare(a, b, reverse):
    res = True if a > b else False
    res = not res if reverse else res
    return res


class RIGate:
    def __init__(self, size, batch_size=600):
        self.num_layers = size
        self.lambda_gates = [None] * self.num_layers
        self.grad_gates = [None] * self.num_layers
        self.device = device
        self.hook_handles = []
        self.dic_gate = {}
        self.distance = []
        self.min_distance = [None] * self.num_layers
        self.min_distance_record = []
        self._reset_window()
        self.lr_case = []
        self.window_case = []
        self.max_lambda = [0] * self.num_layers
        self.strength = []
        self.unit_step = [0] * self.num_layers
        self.factor = 2
        self.total_step = batch_size * 0.1
        self.gate_step = batch_size
        self.stabilizer = 0.00005
        self._old_weight = []
        self._new_weight = []
        self.v1 = [None] * self.num_layers
        self.v2 = [None] * self.num_layers
        self.speedup = False
        self.layers_active_status = []

    def register_hook(self, model, names, old_size, new_size, scale_list):
        dic_size = {}

        for key, o, n, s in zip(names, old_size, new_size, scale_list):
            dic_size[key] = (o, n, s)

        i = 0
        for name, module in model.named_modules():
            if name in names:
                if name.startswith('fea'):
                    d = 4
                else:
                    d = 2
                handle = module.register_forward_hook(
                    self.hook_maker(dic_size[name], i, dim=d, fill=dic_size[name][2]))
                i = i + 1
                self.hook_handles.append(handle)

    def register_hook_new(self, model, old_size, new_size, scale_list):
        dic_size = []

        for o, n, s in zip(old_size, new_size, scale_list):
            dic_size.append((o, n, s))

        for i, module in enumerate(find_modules_short(model)):
            d = module.weight.data.dim()
            handle = module.register_forward_hook(self.hook_maker(dic_size[i], i, dim=d, fill=dic_size[i][2]))
            self.hook_handles.append(handle)

    def hook_maker(self, shape, index, dim=4, fill=2, new_fill=1):

        # old scale, new is 1
        if self.lambda_gates[index] is None:
            t_old = torch.empty(shape[0]).fill_(fill)
            t_new = torch.empty(shape[1]).fill_(new_fill)
            t_new[:shape[0]] = t_old
        else:
            t_old = torch.flatten(self.lambda_gates[index] * fill)
            t_new = torch.empty(shape[1]).fill_(new_fill)
            t_new[:shape[0]] = t_old

        if dim == 4:
            t_new = torch.reshape(t_new, (1, -1, 1, 1))
            t_new = t_new.requires_grad_(False)
        else:
            t_new = torch.reshape(t_new, (1, -1))
            t_new = t_new.requires_grad_(False)

        t_new = t_new.to(self.device)

        # initialize lambda and grad gates
        self.lambda_gates[index] = t_new

        # grad gate lambda  and 1
        self.grad_gates[index] = t_new.clone()

        # grad gate lambda # lambda
        # self.grad_gates[index] = t_new.clone().fill_(torch.flatten(t_new)[0])

        # grad gate lambda and 1/lambda
        # self.grad_gates[index] = t_new.clone()
        # self.grad_gates[index][:, shape[0]:] = 1 / torch.flatten(t_new)[0]

        # set max lambda
        self.max_lambda[index] = fill

        self.unit_step[index] = (fill - 1) / self.total_step

        gate = self.lambda_gates[index]

        def hook(self, input, output):
            output.mul_(gate)

        return hook

    def momentum_function(self, v, delta, momentum=0.99):
        new_v = momentum * v - (1 - momentum) * delta
        return new_v

    def remove_hook(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def is_active(self):
        if self.hook_handles:
            return True
        else:
            return False

    # def _release_inhibite(self, layer, old_ri, new_ri, splite_index, step=0, total_step=300, reverse=True,
    #                       is_step=True):
    #     splite = splite_index[layer]
    #     # current ri scale for old and new
    #     old_scale = torch.flatten(self.gates[layer])[0]
    #     new_scale = torch.flatten(self.gates[layer])[-1]
    #
    #     # ri factor for new and old
    #     distance = old_scale - new_scale
    #     step_left = total_step - step
    #     d_step = distance / step_left
    #
    #     # split d_step according to ri
    #     if reverse:
    #         old_d = d_step / (old_ri + new_ri) * old_ri
    #         new_d = d_step / (old_ri + new_ri) * new_ri
    #     else:
    #         old_d = d_step / (old_ri + new_ri) * new_ri
    #         new_d = d_step / (old_ri + new_ri) * old_ri
    #
    #     # update gate
    #     if is_step:
    #         self.gates[layer][:, :splite] = self.gates[layer][:, :splite] - old_d
    #         self.gates[layer][:, splite:] = self.gates[layer][:, splite:] + new_d
    #     else:
    #         self.gates[layer][:, :splite] = self.gates[layer][:, :splite] - old_ri
    #         self.gates[layer][:, splite:] = self.gates[layer][:, splite:] + new_ri

    def _ri_window(self, old_delta, new_delta, step=0, window_size=10, is_window=False):
        # first step reset the window
        if step == 0:
            self._reset_window()

        if len(self.delta_s_window['old']) == window_size:
            self.delta_s_window['old'].popleft()
            self.delta_s_window['old'].append(old_delta)

            self.delta_s_window['new'].popleft()
            self.delta_s_window['new'].append(new_delta)

            return np.mean(self.delta_s_window['old']), np.mean(self.delta_s_window['new'])

        elif len(self.delta_s_window['old']) == window_size - 1:
            self.delta_s_window['old'].append(old_delta)
            self.delta_s_window['new'].append(new_delta)
            old_mean, new_mean = np.mean(self.delta_s_window['old']), np.mean(self.delta_s_window['new'])
            if is_window and len(self.delta_s_window['old']) == window_size:
                self._reset_window()
            return old_mean, new_mean

        else:
            self.delta_s_window['old'].append(old_delta)
            self.delta_s_window['new'].append(new_delta)

            return None, None

    def _reset_window(self):
        self.delta_s_window = {'old': deque([]), 'new': deque([]), }

    def _ri_unit(self, layer, step, total_step):
        old_scale = torch.flatten(self.grad_gates[layer])[0]
        new_scale = torch.flatten(self.grad_gates[layer])[-1]
        distance = torch.abs(old_scale - new_scale)
        step_left = total_step - step
        if step_left > 100:
            step_left = 100

        if step_left < 50:
            step_left = 50
        d_step = distance / step_left
        return d_step
        # return self.unit_step[layer]

    def _release_inhibite(self, layer, exist_w, distance, splite_index, step=0, total_step=600):
        splite = splite_index[layer]
        # current ri scale for old and new
        # old_scale = torch.flatten(self.lambda_gates[layer])[0]
        # new_scale = torch.flatten(self.lambda_gates[layer])[-1]

        # calculate delta lambda
        left_step = self.gate_step - step
        unit_step = distance / left_step
        delta_lambda = unit_step / exist_w

        self.lambda_gates[layer][:, :splite] = self.lambda_gates[layer][:, :splite] - delta_lambda

        old_scale = torch.flatten(self.lambda_gates[layer])[0]

        if old_scale > self.max_lambda[layer]:
            self.lambda_gates[layer][:, :splite] = self.max_lambda[layer]
        elif old_scale < 1:
            self.lambda_gates[layer][:, :splite] = 1

        self.lambda_gates[layer][:, splite:] = self.lambda_gates[layer][:, splite:]

    def _release_inhibite_auto(self, layer, old_ri, new_ri, splite_index, step=0, total_step=300):

        splite = splite_index[layer]
        # current ri scale for old and new
        old_scale = torch.flatten(self.lambda_gates[layer])[0]
        new_scale = torch.flatten(self.lambda_gates[layer])[-1]

        distance = old_scale - new_scale
        left_step = self.gate_step - step
        decay_step = distance / left_step

        total = torch.flatten(self.lambda_gates[layer]).shape[0]
        old = splite_index[layer]
        new = total - old

        old_decay = decay_step / (old + new) * old
        new_decay = decay_step / (old + new) * new

        self.lambda_gates[layer][:, :splite] = self.lambda_gates[layer][:, :splite] - old_decay
        self.lambda_gates[layer][:, splite:] = self.lambda_gates[layer][:, splite:] + new_decay

    def _release_inhibite_decay(self, layer, old_ri, new_ri, splite_index, step=0, total_step=600):

        splite = splite_index[layer]
        # current ri scale for old and new
        old_scale = torch.flatten(self.lambda_gates[layer])[0]
        # new_scale = torch.flatten(self.lambda_gates[layer])[-1]

        # distance = old_scale - new_scale
        distance = old_scale - 1
        left_step = total_step - step
        decay_step = distance / left_step

        # total = torch.flatten(self.lambda_gates[layer]).shape[0]
        # old = splite_index[layer]
        # new = total - old

        # old_decay = decay_step / (old + new) * old
        # new_decay = decay_step / (old + new) * new

        self.lambda_gates[layer][:, :splite] = self.lambda_gates[layer][:, :splite] - decay_step
        # self.lambda_gates[layer][:, splite:] = self.lambda_gates[layer][:, splite:] + new_decay

    def _release_inhibite_grad(self, layer, old_ri, new_ri, splite_index, step=0, total_step=300, reverse=True,
                               is_step=True):
        splite = splite_index[layer]
        # current ri scale for old and new
        old_scale = torch.flatten(self.grad_gates[layer])[0]
        new_scale = torch.flatten(self.grad_gates[layer])[-1]

        # ri factor for new and old
        distance = old_scale - new_scale
        step_left = total_step - step
        d_step = distance / step_left

        # split d_step according to ri
        if not is_step:
            old_ri = d_step / (old_ri + new_ri) * old_ri
            new_ri = d_step / (old_ri + new_ri) * new_ri

        # update gate
        if not reverse:
            self.grad_gates[layer][:, :splite] = self.grad_gates[layer][:, :splite] + old_ri
            self.grad_gates[layer][:, splite:] = self.grad_gates[layer][:, splite:] + new_ri
        else:
            self.grad_gates[layer][:, :splite] = self.grad_gates[layer][:, :splite] + new_ri
            self.grad_gates[layer][:, splite:] = self.grad_gates[layer][:, splite:] + old_ri

        old_scale = torch.flatten(self.grad_gates[layer])[0]
        new_scale = torch.flatten(self.grad_gates[layer])[-1]

        high_bound = self.max_lambda[layer] * self.factor
        # high_bound = self.max_lambda[layer] * 2
        # low_bound = 0.1
        low_bound = 1 / high_bound

        if old_scale > high_bound:
            self.grad_gates[layer][:, :splite] = high_bound

        elif old_scale < low_bound:
            self.grad_gates[layer][:, :splite] = low_bound

        if new_scale < low_bound:
            self.grad_gates[layer][:, splite:] = low_bound

        elif new_scale > high_bound:
            self.grad_gates[layer][:, splite:] = high_bound

    def _spllit_unit_step(self, unit, s1, s2, is_reverse=False):
        s1 = abs(s1)
        s2 = abs(s2)
        out1 = unit / (s1 + s2) * s1
        out2 = unit / (s1 + s2) * s2
        if is_reverse:
            return out2, out1
        else:
            return out1, out2

    def step_v1(self, pre_s, cur_s, pre_w, cur_w, splite_index, step=0, total_step=300):

        layer = 0

        # switch map
        case_switch = {
            # compare
            True: {
                # direction
                True: {
                    # direction
                    True: 1,
                    False: 5,
                },
                False: {
                    True: 6,
                    False: 4,
                },

            },

            False: {
                True: {
                    True: 2,
                    False: 5,
                },
                False: {
                    True: 6,
                    False: 3,
                },

            },
        }

        for pre, cur, p_w, c_w, splite in zip(pre_s, cur_s, pre_w, cur_w, splite_index):
            # break when delta is not available
            if len(pre) != len(cur) or len(cur) <= splite or len(pre) <= splite:
                break

            if len(p_w) != len(c_w) or len(p_w) <= splite or len(c_w) <= splite:
                break

            # prepare lambda gate decay
            old_cur_w = c_w[:splite]
            old_pre_w = p_w[:splite]
            new_cur_w = c_w[splite:]
            new_pre_w = p_w[splite:]

            mean_old_w = np.mean(old_cur_w)
            mean_new_w = np.mean(new_cur_w)
            old_delta_w = np.mean(old_cur_w) - np.mean(old_pre_w)
            new_delta_w = np.mean(new_cur_w) - np.mean(new_pre_w)
            old_scale = torch.flatten(self.lambda_gates[layer])[0]
            dis = mean_old_w * old_scale - mean_new_w

            # lambda decay
            # self._release_inhibite(layer, mean_old_w, dis, splite_index, step, total_step)
            self._release_inhibite_auto(layer, old_delta_w, new_delta_w, splite_index, step, total_step)

            # prepare
            old_cur = cur[:splite]
            old_pre = pre[:splite]
            new_cur = cur[splite:]
            new_pre = pre[splite:]
            old_mean = np.mean(old_cur)
            new_mean = np.mean(new_cur)
            unit_step = self._ri_unit(layer, step, total_step)
            # unit_step = self.unit_step[layer]

            # minus then mean
            # old_delta = np.mean(old_cur-old_pre)
            # new_delta = np.mean(new_cur-new_pre)

            # mean then minus
            old_delta = np.mean(old_cur) - np.mean(old_pre)
            new_delta = np.mean(new_cur) - np.mean(new_pre)

            # window delta
            old_wind_delta, new_wind_delta = self._ri_window(old_delta, new_delta, step=step, window_size=50,
                                                             is_window=True)

            # is reversely operate
            reverse = False if old_mean >= new_mean else True

            # delta comparison, direction
            delta_compare = _ri_compare(abs(old_delta), abs(new_delta), reverse)
            old_direction = _ri_compare(old_delta, 0, reverse)
            new_direction = _ri_compare(new_delta, 0, reverse)

            # window delta case
            if old_wind_delta is not None:
                delta_wind_compare = _ri_compare(abs(old_wind_delta), abs(new_wind_delta), reverse)
                window_case = case_switch[delta_wind_compare][old_direction][new_direction]
            else:
                window_case = 0

            # lr speed-up case
            lr_case = case_switch[delta_compare][old_direction][new_direction]
            window_case = case_switch[delta_compare][old_direction][new_direction]

            # strength of unit
            if not reverse:
                strength = old_mean / (new_mean + self.stabilizer)
            else:
                strength = new_mean / (old_mean + self.stabilizer)

            if strength > self.max_lambda[layer]:
                strength = self.max_lambda[layer].tolist()

            # distance and update min distance
            distance = np.abs(new_mean - old_mean)
            if self.min_distance[layer] is None or self.min_distance[layer] > distance:
                self.min_distance[layer] = distance

            # save record for analysis
            self.distance.append(distance)
            self.min_distance_record.append(self.min_distance[layer])
            self.window_case.append(window_case)
            self.lr_case.append(lr_case)
            self.strength.append(strength)
            old_scale = torch.flatten(self.lambda_gates[layer])[0].tolist()
            self._old_weight.append(old_delta_w * old_scale)
            self._new_weight.append(new_delta_w)
            # layer += 1
            #     continue

            # RI algorithm
            if distance < self.min_distance[layer]:
                # self._release_inhibite(layer, 1, 0, splite_index, step=step, total_step=total_step)
                pass
            else:
                if window_case == 1:
                    out1, out2 = self._spllit_unit_step(unit_step, old_delta, new_delta)
                    self._release_inhibite_grad(layer, out1 * strength, out2 * strength, splite_index, step, total_step,
                                                reverse=reverse, is_step=True)

                    # self._release_inhibite(layer, 0, unit_step * strength, splite_index, step, total_step,
                    #                        reverse=reverse, is_step=True)
                elif window_case == 2:
                    out1, out2 = self._spllit_unit_step(unit_step, old_delta, new_delta)
                    self._release_inhibite_grad(layer, out1 * strength, out2 * strength, splite_index, step, total_step,
                                                reverse=reverse, is_step=True)
                elif window_case == 3:
                    out1, out2 = self._spllit_unit_step(unit_step, old_delta, new_delta, is_reverse=True)
                    self._release_inhibite_grad(layer, -out1 * strength, -out2 * strength, splite_index, step,
                                                total_step, reverse=reverse, is_step=True)
                    # self._release_inhibite_grad(layer, 0, -unit_step * strength, splite_index, step, total_step,
                    #                             reverse=reverse, is_step=True)
                    # self._release_inhibite(layer, 0, unit_step * strength, splite_index, step, total_step,
                    #                        reverse=reverse, is_step=True)
                elif window_case == 4:
                    out1, out2 = self._spllit_unit_step(unit_step, old_delta, new_delta, is_reverse=True)
                    self._release_inhibite_grad(layer, -out1 * strength, -out2 * strength, splite_index, step,
                                                total_step, reverse=reverse, is_step=True)
                    # self._release_inhibite_grad(layer, 0, -unit_step * strength, splite_index, step, total_step,
                    #                             reverse=reverse, is_step=True)
                elif window_case == 5:
                    self._release_inhibite_grad(layer, unit_step * strength, -unit_step * strength, splite_index, step,
                                                total_step, reverse=reverse, is_step=True)
                    # self._release_inhibite(layer, -unit_step * strength, unit_step * strength, splite_index, step,
                    #                        total_step, reverse=reverse, is_step=True)
                elif window_case == 6:
                    self._release_inhibite_grad(layer, -unit_step * strength, unit_step * strength,
                                                splite_index, step, total_step, reverse=False, is_step=True)
                else:
                    # do nothing as the window case is None
                    pass

            layer += 1

    def step_without_grow(self, splite_index, step=0):
        for layer, splite in enumerate(splite_index):
            self._release_inhibite_grad_auto(layer, splite, step=step, total_step=self.gate_step)

    def step(self, pre_s, cur_s, pre_w, cur_w, splite_index, step=0):
        if step < 3:
            return 0

        # switch map
        case_switch = {
            # compare
            True: {
                # direction
                True: {
                    # direction
                    True: 1,
                    False: 5,
                },
                False: {
                    True: 6,
                    False: 4,
                },

            },

            False: {
                True: {
                    True: 2,
                    False: 5,
                },
                False: {
                    True: 6,
                    False: 3,
                },

            },
        }

        for layer, z in enumerate(zip(pre_s, cur_s, pre_w, cur_w, splite_index)):
            pre, cur, p_w, c_w, splite = z
            # break when delta is not available
            # if len(pre) != len(cur) or len(cur) <= splite or len(pre) <= splite:
            #     continue
            #
            # if len(p_w) != len(c_w) or len(p_w) <= splite or len(c_w) <= splite:
            #     continue

            # prepare lambda gate decay
            old_cur_w = c_w[:splite]
            old_pre_w = p_w[:splite]

            new_cur_w = c_w[splite:]
            if not new_cur_w:
                new_cur_w.append(0)

            new_pre_w = p_w[splite:]
            if not new_pre_w:
                new_pre_w.append(0)

            mean_old_w = np.mean(old_cur_w)
            mean_new_w = np.mean(new_cur_w)
            old_delta_w = np.mean(old_cur_w) - np.mean(old_pre_w)
            new_delta_w = np.mean(new_cur_w) - np.mean(new_pre_w)
            old_scale = torch.flatten(self.lambda_gates[layer])[0]
            dis = mean_old_w * old_scale - mean_new_w

            # prepare
            old_cur = cur[:splite]
            old_pre = pre[:splite]

            new_cur = cur[splite:]
            if not new_cur:
                new_cur.append(0)

            new_pre = pre[splite:]
            if not new_pre:
                new_pre.append(0)

            old_mean = np.mean(old_cur)
            new_mean = np.mean(new_cur)
            old_pre_mean = np.mean(old_pre)
            new_pre_mean = np.mean(new_pre)
            unit_step = self._ri_unit(layer, step, self.gate_step)
            # unit_step = self.unit_step[layer]

            # minus then mean
            # old_delta = np.mean(old_cur-old_pre)
            # new_delta = np.mean(new_cur-new_pre)

            # mean then minus
            old_delta = np.mean(old_cur) - np.mean(old_pre)
            new_delta = np.mean(new_cur) - np.mean(new_pre)

            # window delta
            old_wind_delta, new_wind_delta = self._ri_window(old_delta, new_delta, step=step, window_size=50,
                                                             is_window=True)

            # is reversely operate
            reverse = False if old_mean >= new_mean else True

            # delta comparison, direction
            delta_compare = _ri_compare(abs(old_delta), abs(new_delta), reverse)
            old_direction = _ri_compare(old_delta, 0, reverse)
            new_direction = _ri_compare(new_delta, 0, reverse)

            # window delta case
            if old_wind_delta is not None:
                delta_wind_compare = _ri_compare(abs(old_wind_delta), abs(new_wind_delta), reverse)
                window_case = case_switch[delta_wind_compare][old_direction][new_direction]
            else:
                window_case = 0

            # lr speed-up case
            lr_case = case_switch[delta_compare][old_direction][new_direction]
            window_case = case_switch[delta_compare][old_direction][new_direction]

            # strength of unit
            if not reverse:
                strength = old_mean / (new_mean + self.stabilizer)
            else:
                strength = new_mean / (old_mean + self.stabilizer)

            if strength > self.max_lambda[layer]:
                strength = self.max_lambda[layer].tolist()

            # distance and update min distance
            distance = np.abs(new_mean - old_mean)
            last_distance = np.abs(new_pre_mean - old_pre_mean)
            delta_distance = abs(distance - last_distance)
            if self.min_distance[layer] is None or self.min_distance[layer] > distance:
                self.min_distance[layer] = distance

            # save record for analysis
            self.distance.append(distance)
            self.min_distance_record.append(self.min_distance[layer])
            self.window_case.append(window_case)
            self.lr_case.append(lr_case)
            self.strength.append(strength)
            old_scale = torch.flatten(self.lambda_gates[layer])[0].tolist()
            self._old_weight.append(old_delta_w * old_scale)
            self._new_weight.append(new_delta_w)
            if not self.layers_active_status[layer]:
                continue

            #
            # decay slowly
            # self._release_inhibite(layer, mean_old_w, dis, splite_index, step)

            # lambda decay
            # self._release_inhibite_decay(layer, old_delta_w, new_delta_w, splite_index, step, self.gate_step)

            # decay R
            self._release_inhibite_auto(layer, old_delta_w, new_delta_w, splite_index, step, self.gate_step)

            # momentum velocity
            # self.velocity_calculation_distance(layer, delta_distance)
            self.velocity_calculation(layer, old_delta, new_delta, reverse)

            # RI algorithm
            if distance < self.min_distance[layer]:
                # self._release_inhibite(layer, 1, 0, splite_index, step=step, total_step=total_step)
                pass
            else:
                if window_case == 1:
                    out1, out2 = self._spllit_unit_step(unit_step, old_delta, new_delta)
                    self._release_inhibite_grad(layer, self.v1[layer] * strength, self.v2[layer] * strength,
                                                splite_index, step, self.gate_step,
                                                reverse=reverse, is_step=True)

                    # self._release_inhibite(layer, 0, unit_step * strength, splite_index, step, total_step,
                    #                        reverse=reverse, is_step=True)
                elif window_case == 2:
                    out1, out2 = self._spllit_unit_step(unit_step, old_delta, new_delta)
                    self._release_inhibite_grad(layer, self.v1[layer] * strength, self.v2[layer] * strength,
                                                splite_index, step, self.gate_step,
                                                reverse=reverse, is_step=True)
                elif window_case == 3:
                    out1, out2 = self._spllit_unit_step(unit_step, old_delta, new_delta, is_reverse=True)
                    self._release_inhibite_grad(layer, -self.v1[layer] * strength, -self.v2[layer] * strength,
                                                splite_index, step,
                                                self.gate_step, reverse=reverse, is_step=True)
                    # self._release_inhibite_grad(layer, 0, -unit_step * strength, splite_index, step, total_step,
                    #                             reverse=reverse, is_step=True)
                    # self._release_inhibite(layer, 0, unit_step * strength, splite_index, step, total_step,
                    #                        reverse=reverse, is_step=True)
                elif window_case == 4:
                    out1, out2 = self._spllit_unit_step(unit_step, old_delta, new_delta, is_reverse=True)
                    self._release_inhibite_grad(layer, -self.v1[layer] * strength, -self.v2[layer] * strength,
                                                splite_index, step,
                                                self.gate_step, reverse=reverse, is_step=True)
                    # self._release_inhibite_grad(layer, 0, -unit_step * strength, splite_index, step, total_step,
                    #                             reverse=reverse, is_step=True)
                elif window_case == 5:
                    self._release_inhibite_grad(layer, self.v1[layer] * strength, -self.v2[layer] * strength,
                                                splite_index, step,
                                                self.gate_step, reverse=reverse, is_step=True)
                    # self._release_inhibite(layer, -unit_step * strength, unit_step * strength, splite_index, step,
                    #                        total_step, reverse=reverse, is_step=True)
                elif window_case == 6:
                    self._release_inhibite_grad(layer, -self.v1[layer] * strength, self.v2[layer] * strength,
                                                splite_index, step, self.gate_step, reverse=False, is_step=True)
                else:
                    # do nothing as the window case is None
                    pass

            if window_case in [3, 4, 6]:
                self.speedup = True
            else:
                self.speedup = False

        # if step == self.gate_step - 1:
        #     self.reset_grad_gate()
        #     self.reset()

    def _release_inhibite_grad_auto(self, layer, splite, step=0, total_step=300):

        # current ri scale for old and new
        old_scale = torch.flatten(self.grad_gates[layer])[0]
        new_scale = torch.flatten(self.grad_gates[layer])[-1]

        old_distance = old_scale - 1
        new_distance = new_scale - 1
        if old_distance == 0 or new_distance == 0:
            return 0
        left_step = total_step - step
        old_decay = old_distance / left_step
        new_decay = new_distance / left_step

        self.grad_gates[layer][:, :splite] = self.grad_gates[layer][:, :splite] - old_decay
        self.grad_gates[layer][:, splite:] = self.grad_gates[layer][:, splite:] - new_decay

    def step_v3(self, pre_s, cur_s, pre_w, cur_w, splite_index, step=0, total_step=300):

        layer = 0

        # switch map
        case_switch = {
            # compare
            True: {
                # direction
                True: {
                    # direction
                    True: 1,
                    False: 5,
                },
                False: {
                    True: 6,
                    False: 4,
                },

            },

            False: {
                True: {
                    True: 2,
                    False: 5,
                },
                False: {
                    True: 6,
                    False: 3,
                },

            },
        }

        for pre, cur, p_w, c_w, splite in zip(pre_s, cur_s, pre_w, cur_w, splite_index):
            # break when delta is not available
            if len(pre) != len(cur) or len(cur) <= splite or len(pre) <= splite:
                break

            if len(p_w) != len(c_w) or len(p_w) <= splite or len(c_w) <= splite:
                break

            # prepare lambda gate decay
            old_cur_w = c_w[:splite]
            old_pre_w = p_w[:splite]
            new_cur_w = c_w[splite:]
            new_pre_w = p_w[splite:]

            mean_old_w = np.mean(old_cur_w)
            mean_new_w = np.mean(new_cur_w)
            old_delta_w = np.mean(old_cur_w) - np.mean(old_pre_w)
            new_delta_w = np.mean(new_cur_w) - np.mean(new_pre_w)
            old_scale = torch.flatten(self.lambda_gates[layer])[0]
            dis = mean_old_w * old_scale - mean_new_w

            # lambda decay
            # self._release_inhibite(layer, mean_old_w, dis, splite_index, step, total_step)
            self._release_inhibite_auto(layer, old_delta_w, new_delta_w, splite_index, step, total_step)

            # prepare
            old_cur = cur[:splite]
            old_pre = pre[:splite]
            new_cur = cur[splite:]
            new_pre = pre[splite:]
            old_mean = np.mean(old_cur)
            new_mean = np.mean(new_cur)
            old_pre_mean = np.mean(old_pre)
            new_pre_mean = np.mean(new_pre)
            unit_step = self._ri_unit(layer, step, total_step)
            # unit_step = self.unit_step[layer]

            # minus then mean
            # old_delta = np.mean(old_cur-old_pre)
            # new_delta = np.mean(new_cur-new_pre)

            # mean then minus
            old_delta = np.mean(old_cur) - np.mean(old_pre)
            new_delta = np.mean(new_cur) - np.mean(new_pre)

            # window delta
            old_wind_delta, new_wind_delta = self._ri_window(old_delta, new_delta, step=step, window_size=50,
                                                             is_window=True)

            # is reversely operate
            reverse = False if old_mean >= new_mean else True

            # delta comparison, direction
            delta_compare = _ri_compare(abs(old_delta), abs(new_delta), reverse)
            old_direction = _ri_compare(old_delta, 0, reverse)
            new_direction = _ri_compare(new_delta, 0, reverse)

            # window delta case
            if old_wind_delta is not None:
                delta_wind_compare = _ri_compare(abs(old_wind_delta), abs(new_wind_delta), reverse)
                window_case = case_switch[delta_wind_compare][old_direction][new_direction]
            else:
                window_case = 0

            # lr speed-up case
            lr_case = case_switch[delta_compare][old_direction][new_direction]
            window_case = case_switch[delta_compare][old_direction][new_direction]

            # trigger lr
            # if lr_case in [2, 4, 6]:
            #     # lr_trigger
            #     pass

            # strength of unit
            if not reverse:
                strength = old_mean / (new_mean + self.stabilizer)
            else:
                strength = new_mean / (old_mean + self.stabilizer)

            if strength > self.max_lambda[layer]:
                strength = self.max_lambda[layer].tolist()

            # distance and update min distance
            distance = np.abs(new_mean - old_mean)
            last_distance = np.abs(new_pre_mean - old_pre_mean)
            delta_distance = distance - last_distance

            if self.min_distance[layer] is None or self.min_distance[layer] > distance:
                self.min_distance[layer] = distance

            # save record for analysis
            self.distance.append(distance)
            self.min_distance_record.append(self.min_distance[layer])
            self.window_case.append(window_case)
            self.lr_case.append(lr_case)
            self.strength.append(strength)
            old_scale = torch.flatten(self.lambda_gates[layer])[0].tolist()
            self._old_weight.append(old_delta_w * old_scale)
            self._new_weight.append(new_delta_w)
            # layer += 1
            #     continue

            # momentum velocity
            self.velocity_calculation_distance(layer, delta_distance)

            # RI algorithm
            if distance < self.min_distance[layer]:
                # self._release_inhibite(layer, 1, 0, splite_index, step=step, total_step=total_step)
                pass
            else:
                if window_case == 1:
                    out1, out2 = self._spllit_unit_step(self.v1[layer], old_delta, new_delta)
                    self._release_inhibite_grad(layer, out1 * strength, out2 * strength, splite_index, step, total_step,
                                                reverse=reverse, is_step=True)

                    # self._release_inhibite(layer, 0, unit_step * strength, splite_index, step, total_step,
                    #                        reverse=reverse, is_step=True)
                elif window_case == 2:
                    out1, out2 = self._spllit_unit_step(self.v1[layer], old_delta, new_delta)
                    self._release_inhibite_grad(layer, out1 * strength, out2 * strength, splite_index, step, total_step,
                                                reverse=reverse, is_step=True)
                elif window_case == 3:
                    out1, out2 = self._spllit_unit_step(self.v1[layer], old_delta, new_delta, is_reverse=True)
                    self._release_inhibite_grad(layer, -out1 * strength, -out2 * strength, splite_index, step,
                                                total_step, reverse=reverse, is_step=True)
                    # self._release_inhibite_grad(layer, 0, -unit_step * strength, splite_index, step, total_step,
                    #                             reverse=reverse, is_step=True)
                    # self._release_inhibite(layer, 0, unit_step * strength, splite_index, step, total_step,
                    #                        reverse=reverse, is_step=True)
                elif window_case == 4:
                    out1, out2 = self._spllit_unit_step(self.v1[layer], old_delta, new_delta, is_reverse=True)
                    self._release_inhibite_grad(layer, -out1 * strength, -out2 * strength, splite_index, step,
                                                total_step, reverse=reverse, is_step=True)
                    # self._release_inhibite_grad(layer, 0, -unit_step * strength, splite_index, step, total_step,
                    #                             reverse=reverse, is_step=True)
                elif window_case == 5:
                    out1, out2 = self._spllit_unit_step(self.v1[layer], old_delta, new_delta)
                    self._release_inhibite_grad(layer, out1 * strength, -out2 * strength, splite_index, step,
                                                total_step, reverse=reverse, is_step=True)
                    # self._release_inhibite(layer, -unit_step * strength, unit_step * strength, splite_index, step,
                    #                        total_step, reverse=reverse, is_step=True)
                elif window_case == 6:
                    self._release_inhibite_grad(layer, -self.v1[layer] * strength, self.v1[layer] * strength,
                                                splite_index, step, total_step, reverse=False, is_step=True)

                else:
                    # do nothing as the window case is None
                    pass

            layer += 1

    def record_gate(self, epoch, batch):
        check_dic_key(self.dic_gate, epoch)
        check_dic_key(self.dic_gate[epoch], batch)
        for layer, gate in enumerate(self.lambda_gates):
            check_dic_key(self.dic_gate[epoch][batch], layer)

            lambda_tensor = torch.flatten(gate)
            list_from_lambda = lambda_tensor.tolist()
            self.dic_gate[epoch][batch][layer]['gate'] = str(list_from_lambda)

            grad_tensor = torch.flatten(self.grad_gates[layer])
            list_from_grad = grad_tensor.tolist()
            self.dic_gate[epoch][batch][layer]['grad_gate'] = str(list_from_grad)

            if self.min_distance_record:
                self.dic_gate[epoch][batch][layer]['min_distance'] = str(
                    self.min_distance_record[layer - self.num_layers])
            else:
                self.dic_gate[epoch][batch][layer]['min_distance'] = str(0)

            if self.distance:
                self.dic_gate[epoch][batch][layer]['distance'] = str(self.distance[layer - self.num_layers])
            else:
                self.dic_gate[epoch][batch][layer]['distance'] = str(0)

            if self.lr_case:
                self.dic_gate[epoch][batch][layer]['lr_case'] = str(self.lr_case[layer - self.num_layers])
            else:
                self.dic_gate[epoch][batch][layer]['lr_case'] = str(0)

            if self.window_case:
                self.dic_gate[epoch][batch][layer]['window_case'] = str(self.window_case[layer - self.num_layers])
            else:
                self.dic_gate[epoch][batch][layer]['window_case'] = str(0)

            if self.strength:
                self.dic_gate[epoch][batch][layer]['strength'] = str(self.strength[layer - self.num_layers])
            else:
                self.dic_gate[epoch][batch][layer]['strength'] = str(0)

            if self._old_weight:
                self.dic_gate[epoch][batch][layer]['old_weight'] = str(self._old_weight[layer - self.num_layers])
            else:
                self.dic_gate[epoch][batch][layer]['old_weight'] = str(0)

            if self._new_weight:
                self.dic_gate[epoch][batch][layer]['new_weight'] = str(self._new_weight[layer - self.num_layers])
            else:
                self.dic_gate[epoch][batch][layer]['new_weight'] = str(0)

    def put_gate_back(self, model):
        layers = find_modules_short(model)
        for i, layer in enumerate(layers[:-1]):
            # gate = self.gates[i]
            dim = layer.weight.data.dim()
            if dim == 4:
                gate = torch.reshape(self.lambda_gates[i], (-1, 1, 1, 1))
            else:
                gate = torch.reshape(self.lambda_gates[i], (-1, 1))
            # scale = torch.flatten(gate)[0]
            layer.weight.data = layer.weight.data * gate
            bias_gate = torch.flatten(gate)
            layer.bias.data = layer.bias.data * bias_gate

            # reset lambda and grad gates
            self.lambda_gates[i].fill_(1)
            self.grad_gates[i].fill_(1)

    def velocity_calculation(self, layer, old_delta, new_delta, reverse):

        if not reverse:
            new_delta = -1 * new_delta
        else:
            old_delta = -1 * old_delta

        # velocity
        if self.v1[layer] is None:
            self.v1[layer] = self.max_lambda[layer] * self.unit_step[layer]
        else:
            self.v1[layer] = self.momentum_function(self.v1[layer], old_delta)

        if self.v2[layer] is None:
            self.v2[layer] = self.unit_step[layer]
        else:
            self.v2[layer] = self.momentum_function(self.v2[layer], new_delta)

    def velocity_calculation_distance(self, layer, distance_delta):

        # velocity
        if self.v1[layer] is None:
            self.v1[layer] = self.unit_step[layer]
        else:
            self.v1[layer] = self.momentum_function(self.v1[layer], distance_delta)

        if self.v1[layer] < 0:
            print('', end='')

    def reset_grad_gate(self):
        for i in range(len(self.grad_gates)):
            self.grad_gates[i].fill_(1)

    def get_dic_gate(self):
        return self.dic_gate

    def get_gates(self):
        return self.lambda_gates

    def gradient_decay(self, model, ex_index):
        layers = find_modules_short(model)
        for i, layer in enumerate(layers[:-1]):
            if not self.layers_active_status[i]:
                continue
            dim = layer.weight.grad.data.dim()
            shape = layer.weight.grad.data.shape[0]
            if dim == 4:
                grad_lambda = torch.reshape(self.grad_gates[i], (-1, 1, 1, 1))
            else:
                grad_lambda = torch.reshape(self.grad_gates[i], (-1, 1))
            layer.weight.grad.data = layer.weight.grad.data / grad_lambda
            grad_lambda = torch.flatten(grad_lambda)
            # r
            # if self.speedup:
            #     r = 1 / ((shape - ex_index[i]) / shape)
            #     layer.bias.grad.data = layer.bias.grad.data / grad_lambda * r
            # else:
            if layer.bias is not None:
                layer.bias.grad.data = layer.bias.grad.data / grad_lambda
        return model

    def gradient_decay_lambda(self, model):
        layers = find_modules_short(model)
        for i, layer in enumerate(layers[:-1]):
            dim = layer.weight.grad.data.dim()
            if dim == 4:
                grad_lambda = torch.reshape(self.lambda_gates[i], (-1, 1, 1, 1))
            else:
                grad_lambda = torch.reshape(self.lambda_gates[i], (-1, 1))
            layer.weight.grad.data = layer.weight.grad.data / grad_lambda
            grad_lambda = torch.flatten(grad_lambda)
            layer.bias.grad.data = layer.bias.grad.data / grad_lambda

    def reset(self):
        self.v1 = [None] * self.num_layers
        self.v2 = [None] * self.num_layers
        self.min_distance = [None] * self.num_layers

    def save_gate_dic_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.dic_gate, f, indent=4)

    def set_layer_status(self, layer_status):
        self.layers_active_status = layer_status
