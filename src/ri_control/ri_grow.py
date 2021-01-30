import numpy as np
import copy
from model.CNN import LeNet5_GROW
from main import generate_data_loader
from utils import dir_path, write_log
import torch
import math
# from memory_profiler import profile

from grow.growth_utils import find_modules_short
from ri_control.ri_regression import svr_regression
from ri_control.flops import calculate_flops, calculate_total_flops
from ri_control.ri_regression import fit_decay, fit_decay_short, fit_self_decay

np.set_printoptions(precision=4)

# v1
e_factor_switch = False
rolling_update = False
layer_switch = True

# v2
# e_factor_switch = True
# rolling_update = True
# layer_switch = False

#########################
# class DynamicMemory:
#     def __init__(self, num_epoch, num_n):
#         self.array = np.zeros((num_epoch, num_n))
#         self.dic_self_decay = {}  #
#         self.dic_deltaloss = {}
#         self.dic_decay = {}
#         self.seed_n = []
#         self.seed_epoch = []
#         self.decay = 0.1
#
#     def put_values(self, batch_loss, epoch, n):
#         mean = np.mean(batch_loss)
#         self_decay = self.generate_self_decay(batch_loss)
#         delta_loss = np.mean(batch_loss[:100]) - np.mean(batch_loss[-100:])
#         self.array[epoch, n] = mean
#         self.dic_self_decay[n] = self_decay
#         self.dic_deltaloss[n] = delta_loss
#         # this will trigger update seed
#         self.update_seed(n, epoch)
#
#     def update_to(self, epoch):
#         # update the seed to n
#         r = []
#         for n, e in zip(self.seed_n, self.seed_epoch):
#             value = self.array[e, n]
#             r_value = self.calculate_decayed_value(e, epoch, value, self.dic_deltaloss[n], self.dic_self_decay[n])
#             r.append(r_value)
#         return r
#
#     def get_values(self, epoch, n):
#         value_array = self.update_to(epoch)
#         decay = self.generate_decay(value_array)
#         delta_loss = abs(value_array[-2] - value_array[-1]) / (self.seed_n[-1] - self.seed_n[-2])
#         expected_loss = self.calculate_decayed_value(self.seed_n[-1], n, value_array[-1], delta_loss, decay)
#         return expected_loss
#
#     def calculate_decayed_value(self, current, target, current_value, delta_loss, decay):
#         diff = target - current
#         decayed_value_array = [current_value]
#         for i in range(1, diff + 1):
#             value = decayed_value_array[-1] - delta_loss * decay ** i
#             decayed_value_array.append(value)
#         for i, value in enumerate(decayed_value_array):
#             if value < 0:
#                 return decayed_value_array[i - 1]
#         return decayed_value_array[-1]
#
#     def update_seed(self, n, epoch):
#         # update seed
#         if n in self.seed_n:
#             i = self.seed_n.index(n)
#             self.seed_epoch[i] = epoch
#         elif len(self.seed_n) < 3:
#             self.seed_n.append(n)
#             self.seed_epoch.append(epoch)
#         elif len(self.seed_n) == 3:
#             min = np.min(self.seed_n)
#             min_index = self.seed_n.index(min)
#             self.seed_n.remove(min)
#             self.seed_n.append(n)
#             self.seed_epoch.pop(min_index)
#             self.seed_epoch.append(epoch)
#
#     def generate_decay(self, array):
#         if len(array) < 3:
#             return self.decay
#             # the dacay must meet step = 1, now is wrong
#         delta_loss = []
#         i = 1
#         while i < len(array):
#             loss = array[i] - array[i - 1]
#             delta_n = self.seed_n[i] - array[i - 1]
#             d_loss = loss / delta_n
#             delta_loss.append(d_loss)
#             i += 1
#         decay = delta_loss[1] / delta_loss[0]
#         return decay
#
#     def generate_self_decay(self, loss, length=50):
#         # i = 0
#         # array = []
#         # delta_loss = []
#         # while i < len(loss):
#         #     array.append(loss[i])
#         #     i = i + length
#         #
#         # for i in range(len(array) - 1):
#         #     delta = array[i] - array[i + 1]
#         #     if delta > 0:  # only remain the value of delta larger than 0
#         #         delta_loss.append(delta)
#         #
#         # pre_delta_loss = np.asarray(delta_loss[:-1])
#         # past_delta_loss = np.asarray(delta_loss[1:])
#         # decay_delta_loss = np.abs(pre_delta_loss / past_delta_loss)
#         # mean_decay = np.median(decay_delta_loss)
#         #
#         pre = np.mean(loss[:length])
#         next = np.mean(loss[-length:])
#         return next / pre


class DynamicMemory:
    def __init__(self, num_epoch, num_n, num_essence):
        self.array = np.zeros((num_epoch, num_n))
        self.dic_self_decay = {}  #
        self.dic_data = {}
        # self.dic_deltaloss = {}
        self.dic_decay = {}
        self.seed_n = []
        self.seed_epoch = []

        self.step = 0.01
        self.decay = 0.9
        self.c = 100
        self.num_cossim = 0

        self.curve_self = True
        self.cache_size = 5

        self.decay_n = {}

        self.num_essense = num_essence
        self.decay_n_factor = 1

        self.decay_n_x_normal = False

    def put_values(self, batch_loss, epoch, n):
        # update regression function
        # if the regression function exists
        #       refitting the regression, so need to memories the data

        if epoch <= 3:
            # update seed-n
            self.update_seed(n, epoch)
            self.update_dic_data_old(batch_loss, n, epoch)
        else:
            # update dic_data and seed-n
            self.update_dic_data(batch_loss, n, epoch)

        # generate SVR model
        # print('self decay epoch_{}'.format(epoch, n), end='')
        self.generate_self_decay(n)

        # update epoch_n mean
        mean = np.mean(batch_loss)
        self.array[epoch, n] = mean

        # delta loss may not be useful anymore
        # update delta loss
        # delta_loss = np.mean(batch_loss[:100]) - np.mean(batch_loss[-100:])
        # self.dic_deltaloss[n] = delta_loss
        # this will trigger update seed

    def update_dic_data_v1(self, loss, n, e):
        if n not in self.dic_data:
            self.dic_data[n] = []
        self.dic_data[n].append((e, loss))

    def update_dic_data_old(self, loss, n, e):
        if n not in self.dic_data:
            self.dic_data[n] = []

        for size in self.seed_n:
            self.dic_data[size].append((e, loss))

    def update_dic_data(self, loss, size_n, epoch):

        if size_n not in self.dic_data:
            self.dic_data[size_n] = []

        dic_self_decay = {}
        dic_decay_n = {}

        # prepare self-decay & decay-n seed
        if size_n in self.seed_n:
            self_seed = self.seed_n[:-1]
            decay_seed = self.seed_n[1:]
        else:
            self_seed = self.seed_n[1:]
            decay_seed = self.seed_n[2:]
            decay_seed.append(size_n)

        # prepare self decay
        for s in self_seed:
            dic_self_decay[s] = np.mean(self.self_decay_to_epoch(epoch, s))

        # prepare decay n
        for s in decay_seed:
            dic_decay_n[s] = self.get_values(epoch, s)

        # update essense group
        self.update_seed(size_n, epoch)

        # rolling update
        n_data = np.asarray(loss)
        index = len(self.seed_n) - 1
        for size in self.seed_n[::-1]:
            if size_n == size:
                # self.dic_data[size].append((epoch, n_data.tolist()))
                # self.dic_data[size].append((epoch, n_data.tolist()))
                self.dic_data_append(size, (epoch, n_data.tolist()))
            else:
                # rolling update
                if not rolling_update:
                    break
                delta_1 = np.mean(self.dic_data[size][-1][-1]) / np.mean(n_data)
                delta_2 = dic_self_decay[size] / dic_decay_n[self.seed_n[index + 1]]
                if delta_1 >= delta_2:
                    n_loss = n_data * (dic_self_decay[size] / dic_decay_n[self.seed_n[index + 1]])
                    print('delta_1 is {}, delta_2 is {}'.format(delta_1, delta_2))
                    print('size {}, pick new scale'.format(size))
                else:
                    n_loss = n_data
                    print('delta_1 is {}, delta_2 is {}'.format(delta_1, delta_2))
                    print('size {}, pick old scale'.format(size))
                # self.dic_data[size].append((epoch, n_loss.tolist()))
                self.dic_data_append(size, (epoch, n_data.tolist()))
                n_data = n_loss
            index -= 1

    def dic_data_append(self, size, tup):
        if len(self.dic_data[size]) >= self.cache_size:
            del self.dic_data[size][0]
        self.dic_data[size].append(tup)

    # def self_decay_to_epoch(self, epoch, n):
    #     x_start = epoch * self.num_cossim * self.step
    #     x_end = x_start + self.num_cossim * self.step
    #     x_predict = np.arange(x_start, x_end, self.step).reshape(-1, 1)
    #     if self.curve_self:
    #         result = self.dic_self_decay[n](x_predict)
    #     else:
    #         result = self.dic_self_decay[n].predict(x_predict)
    #     return result

    def self_decay_to_epoch(self, epoch, n):
        x_start = epoch
        x_end = epoch + 1
        step = (x_end - x_start) / self.num_cossim
        x_predict = np.arange(x_start, x_end, step).reshape(-1, 1)
        if self.curve_self:
            x_predict = x_predict.reshape(-1)
            result = self.dic_self_decay[n][0](x_predict)
        else:
            result = self.dic_self_decay[n][0].predict(x_predict)
        return result

    def update_to(self, epoch):
        # predict the three regression function to next epoch
        # r = []
        r = None
        # for n, e in zip(self.seed_n, self.seed_epoch):
        #     value = self.array[e, n]
        #     r_value = self.calculate_decayed_value(e, epoch, value, self.dic_deltaloss[n], self.dic_self_decay[n])
        #     r.append(r_value)
        # return r
        for n in self.seed_n:

            # result = self.self_decay_to_epoch(epoch, n)
            # m_result = np.mean(result)
            # r.append(m_result)

            result = self.self_decay_to_epoch(epoch, n)
            if r is None:
                r = result
            else:
                r = np.hstack((r, result))
        # print('value generated by self-decay in Epoch-{} is {}'.format(epoch, r))
        return r

    # def get_values_old(self, epoch, n):
    #     # get estimated score
    #     value_array = self.update_to(epoch)
    #     decay = self.generate_decay(value_array)
    #     delta_loss = abs(value_array[-2] - value_array[-1]) / (self.seed_n[-1] - self.seed_n[-2])
    #     expected_loss = self.calculate_decayed_value(self.seed_n[-1], n, value_array[-1], delta_loss, decay)
    #     return expected_loss

    def get_values(self, epoch, n):
        # pass state and action in
        # get estimated score
        # if self.decay_n[epoch] is None:
        if epoch not in self.decay_n:
            value_array = self.update_to(epoch)
            print('epoch = {}, size ={} : decayN - '.format(epoch, n), end='')
            self.decay_n[epoch] = self.fit_ndecay(value_array)
        # n = n * self.decay_n_factor

        # minus the first seed
        if self.decay_n_x_normal:
            n = n - self.seed_n[0] + 1

        expected_loss = self.decay_n[epoch][0](n)
        # decay = self.generate_decay(value_array)
        # delta_loss = abs(value_array[-2] - value_array[-1]) / (self.seed_n[-1] - self.seed_n[-2])
        # expected_loss = self.calculate_decayed_value(self.seed_n[-1], n, value_array[-1], delta_loss, decay)
        return expected_loss

    def calculate_decayed_value(self, current, target, current_value, delta_loss, decay):
        # calculate the decay value for N-decay
        diff = target - current
        decayed_value_array = [current_value]
        for i in range(1, diff + 1):
            value = decayed_value_array[-1] - delta_loss * decay ** i
            decayed_value_array.append(value)
        for i, value in enumerate(decayed_value_array):
            if value < 0:
                return decayed_value_array[i - 1]
        return decayed_value_array[-1]

    # def fit_ndecay(self, value_array):
    #     x = np.asarray(self.seed_n)
    #     # x = x * self.decay_n_factor
    #     y = np.asarray(value_array)
    #     if len(value_array) < 3:
    #         fitted_curve = fit_decay_short(x, y)
    #     else:
    #         fitted_curve = fit_decay(x, y)
    #     return fitted_curve

    def fit_ndecay(self, value_array):
        sample_rate = len(value_array) // len(self.seed_n)
        x_step = 1 / sample_rate
        x = None
        y = np.asarray(value_array)
        for i in self.seed_n:
            x_i = np.arange(i, i + 1, x_step)
            if x is None:
                x = x_i
            else:
                x = np.hstack((x, x_i))

        if self.decay_n_x_normal:
            x = x - self.seed_n[0] + 1

        if len(self.seed_n) < 3:
            fitted_curve = fit_decay_short(x, y)
        else:
            # fitted_curve = fit_decay(x, y)
            fitted_curve = fit_self_decay(x, y)
        return fitted_curve

    def update_seed(self, n, epoch):
        # maintain three Self-decay
        # update seed
        if n in self.seed_n:
            i = self.seed_n.index(n)
            self.seed_epoch[i] = epoch
        elif len(self.seed_n) < self.num_essense:
            self.seed_n.append(n)
            self.seed_epoch.append(epoch)
        elif len(self.seed_n) == self.num_essense:
            min = np.min(self.seed_n)
            min_index = self.seed_n.index(min)
            self.seed_n.remove(min)
            self.seed_n.append(n)
            self.seed_epoch.pop(min_index)
            self.seed_epoch.append(epoch)

    def generate_decay(self, array):
        # so far we didnt use this function, as two points are still converged
        if len(array) < 3:
            return self.decay
            # the dacay must meet step = 1, now is wrong
        delta_loss = []
        i = 1
        while i < len(array):
            loss = array[i] - array[i - 1]
            delta_n = self.seed_n[i] - array[i - 1]
            d_loss = loss / delta_n
            delta_loss.append(d_loss)
            i += 1
        decay = delta_loss[1] / delta_loss[0]
        return decay

    # def generate_self_decay(self, n):
    #     # generate regression model and save it to the dictionary
    #     if n not in self.dic_data:
    #         raise Exception('Could not find the key in data dictionary')
    #
    #     es = []
    #     losses = []
    #     for epoch_loss in self.dic_data[n]:
    #         e, loss = epoch_loss
    #         es.append(e)
    #         losses += loss
    #
    #     num = len(losses)
    #     if self.num_cossim == 0:
    #         self.num_cossim = num
    #     start = es[0] * num * self.step
    #     end = start + self.step * num
    #     x = np.arange(start, end, self.step).reshape(-1, 1)
    #     losses = np.asarray(losses) * self.c
    #     svr = svr_regression(x, losses)
    #     self.dic_self_decay[n] = svr
    #
    # def generate_self_decay(self, n=0):
    #     # generate regression model and save it to the dictionary
    #     if n not in self.dic_data:
    #         raise Exception('Could not find the key in data dictionary')
    #
    #     for size in self.seed_n:
    #         es = []
    #         losses = []
    #         for epoch_loss in self.dic_data[size]:
    #             e, loss = epoch_loss
    #             es.append(e)
    #             losses += loss
    #
    #         num = len(losses) / len(es)
    #         if self.num_cossim == 0:
    #             self.num_cossim = num
    #         start = es[0] * num * self.step
    #         end = start + self.step * num * len(es)
    #         x = np.arange(start, end, self.step).reshape(-1, 1)
    #
    #         if len(x) > len(losses):
    #             x = x[:len(losses)]
    #         losses = np.asarray(losses) * self.c
    #         if self.curve_self:
    #             x = x.reshape(-1)
    #             losses = losses.reshape(-1)
    #             # svr = fit_decay(x, losses)
    #             print('Self-Decay size_{} :'.format(size), end='')
    #             svr = fit_self_decay(x, losses)
    #         else:
    #             svr = svr_regression(x, losses)
    #         self.dic_self_decay[size] = svr

    def generate_self_decay(self, n=0):
        # generate regression model and save it to the dictionary
        if n not in self.dic_data:
            raise Exception('Could not find the key in data dictionary')

        for size in self.seed_n:
            es = []
            losses = []
            for epoch_loss in self.dic_data[size]:
                e, loss = epoch_loss
                es.append(e)
                losses += loss

            num = len(losses) / len(es)
            if self.num_cossim == 0:
                self.num_cossim = num

            start = es[0]
            end = es[-1] + 1
            step = (end - start) / len(losses)
            x = np.arange(start, end, step).reshape(-1, 1)

            if len(x) > len(losses):
                x = x[:len(losses)]
            losses = np.asarray(losses) * self.c
            if self.curve_self:
                x = x.reshape(-1)
                losses = losses.reshape(-1)
                # svr = fit_decay(x, losses)
                print('Self-Decay size_{} :'.format(size), end='')
                # fit change to constant bound fit decay
                svr = fit_self_decay(x, losses)
                # bound = np.mean(losses) - np.std(losses)
                # svr = fit_decay_constant(x, losses, bound)
            else:
                svr = svr_regression(x, losses)
            self.dic_self_decay[size] = svr


class State:
    def __init__(self, loss, n, epoch, in_shape, next_in_shape, w_shape, next_w_shape, list=[]):
        self.loss = loss
        self.n = n
        self.epoch = epoch
        self.in_shape = in_shape
        self.next_in_shape = next_in_shape
        self.w_shape = w_shape
        self.next_w_shape = next_w_shape
        self.list = list

    def __str__(self):
        return 'size: {},loss: {:.4f},epoch: {}'.format(self.n, self.loss, self.epoch)


def obt_memory_size(weight_shape):
    size = 1
    for i in range(len(weight_shape)):
        size *= weight_shape[i]
    return size


class GrowController:
    model_size = []

    def __init__(self, discount_factor, num_actions, horizon_step, index, cost_factor, cost_base, e_factor=0.4,
                 num_essence=3, log_path=''):  # , init_model, trian, test
        self.horizon_steps = horizon_step
        self.num_actions = num_actions
        self.state = None
        self.obj_loss_mean = DynamicMemory(55, 200, num_essence)
        self.array_estimate_loss = []
        self.discount_factor = discount_factor
        self.expected_loss = []
        self.train_loss = []
        self.epoch = 0
        self.size = 0
        self.index = index
        self.flops_cost = 0
        self.memory_cost = 0
        self.input_shape = []
        self.weight_shape = []
        self.next_input_shape = []
        self.next_weight_shape = []
        self.cost_factor = cost_factor
        self.cost_decay = False
        self.cost_decay_step = 4
        self.decay_n = []
        self.log_path = log_path
        self.cost_base = cost_base
        # self.confidence = 0.2
        # self.co_value = 1
        # self.vertical_decay = 0.95
        self.dropout_rate = 0.5
        self.grow_rate = 0.6
        self.e_factor = e_factor
        self.g_value = None

    def generate_actions(self, state):
        """
        this action set includes the current structure N
        :param state: state passed
        :return:  return an array of actions for the state passed
        """
        # if grow_rate is not 0
        if self.grow_rate > 0:
            self.num_actions = math.floor(self.size * self.grow_rate)

        current_n = state.n
        actions = [i for i in range(current_n, min(current_n + self.num_actions, current_n + self.size + 1))]
        return actions

    def value_function_old(self, state, discount_factor, step=0):
        # state is current statec
        # action is current action
        # action_size
        action_list = []
        cost_list = []
        # if state.loss < self.threshold:
        if step > self.size or step > self.horizon_steps:
            print('', end='\n')
            return self.state.n
        else:
            actions = self.generate_actions(state)
            # the cost for this action as action stands for n
            # estimate loss for each action
            # save the action list for analysis
            estimate_loss = [self.obj_loss_mean.get_values(state.epoch + 1, action) for action in actions]
            estimate_delta_loss = state.loss - np.asarray(estimate_loss)
            # estimate_loss / cost function
            cost = np.asarray([self.obtain_delta_cost(action) for action in actions])
            estimate_value = estimate_delta_loss / cost
            # print essential value
            max_index = np.argmax(estimate_value)
            value = estimate_value[max_index]
            selected_action = actions[max_index]
            state = self.state_transition(state, selected_action, estimate_loss=estimate_loss[max_index])
            for i, a in enumerate(actions):
                print(
                    'a:{},l:{:.5f},d:{:.5f},c:{},v:{:.5f} ||'.format(a, estimate_loss[i], estimate_delta_loss[i],
                                                                     cost[i],
                                                                     estimate_value[i]), end='')
            print('-> A:{},V:{}'.format(selected_action, value), end='\n')
            action_list.append(selected_action)
            return value + discount_factor * self.value_function(state, discount_factor, step + 1)

    # def value_function(self, state, max_step, max_n):
    #     if state.epoch >= max_step:
    #         r, l = self.get_reward(state, state.n)
    #         return r
    #     else:
    #         actions = []
    #         for a in range(state.n, max_n):
    #             _, l = self.get_reward(state, a)
    #             new_state = self.state_transition(state, a, l)  # transfter state
    #             actions.append(self.value_function(new_state, max_step, max_n))
    #
    #         r, _ = self.get_reward(state, state.n)
    #         return r + self.discount_factor * np.mean(actions) + self.confidence * len(actions)
    #         # return r + self.discount_factor * np.max(actions)

    def value_function(self, state, init_state, max_step, max_n, parent):
        if state.epoch >= max_step:
            r, l = self.get_reward(parent, state.n, init_state)
            state.list.append(state.n)
            # max_r, _ = self.get_reward(state, max_n - 1)
            # r = r + self.co_value * (max_r - r)
            return r
        else:
            r, l = self.get_reward(parent, state.n, init_state)
            state.loss = l
            state.list.append(state.n)

            actions = []
            leaf_state = [None] * (max_n - state.n + 1)
            dropout = np.random.binomial(1, self.dropout_rate, self.num_actions)
            for a in range(state.n, max_n + 1):
                # r, l = self.get_reward(state, a)
                if dropout[a - state.n] == 0 and a != state.n:
                    actions.append(-50000)
                    continue
                new_max_n = a + self.num_actions - 1
                leaf_state[a - state.n] = self.state_transition(state, a, l)  # transfter state

                leaf_v = self.value_function(leaf_state[a - state.n], init_state, max_step, new_max_n, state)

                if a != state.n and leaf_v <= actions[a - state.n - 1]:
                    break

                actions.append(leaf_v)

            # max_r, _ = self.get_reward(state, max_n - 1)
            # r = r + self.co_value * (max_r - r)
            # reward = r * (0.9 ** (state.n - init_state.n + state.epoch - init_state.epoch))
            # reward = r * (0.5 ** (state.n - init_state.n ))
            # delta_r = r_last - r - reward
            # print('Epoch is {}, Action size is {}, detla R is {}'.format(state.epoch, state.n, delta_r))
            argmax = np.argmax(actions)
            actionmax = state.n + argmax
            state.list += leaf_state[argmax].list
            # print('state list is {}'.format(state.list))
            # print('leaf state is {}'.format(leaf_state[argmax].list))
            #
            # print('Epoch is {}, Action is {}, max is {}, value is {}'.format(state.epoch, state.n, actionmax, np.max(actions)))

            return r + self.discount_factor * np.max(actions)
            # return r + self.discount_factor * np.mean(actions)
            # return self.discount_factor * np.mean(actions)

    def cost_function(self, state, max_step, max_n):
        if state.epoch >= max_step:
            c, l = self.get_cost(state, state.n)
            return c
        else:
            actions = []
            for a in range(state.n, max_n):
                _, l = self.get_cost(state, a)
                new_state = self.state_transition(state, a, l)  # transfter state
                actions.append(self.cost_function(new_state, max_step, max_n))

            c, _ = self.get_cost(state, state.n)
            return c + self.discount_factor * np.max(actions)
            # return r + self.discount_factor * np.max(actions)

    def delta_function(self, state, max_step, max_n):
        if state.epoch >= max_step:
            c, l = self.get_detla(state, state.n)
            return c
        else:
            c, l = self.get_detla(state, state.n)

            actions = []
            for a in range(state.n, max_n):
                # _, l = self.get_detla(state, a)
                new_state = self.state_transition(state, a, l)  # transfter state
                actions.append(self.delta_function(new_state, max_step, max_n))

            return c + self.discount_factor * np.max(actions)

    # @profile
    def one_step_lookahead(self, state, actions):
        # return the value for each aciton and the
        # num = min(self.num_actions, self.size)
        num = len(actions)
        max_step = state.epoch + self.horizon_steps
        # self.obj_loss_mean.decay_n = [None] * (max_step + 2)

        A = np.zeros(num)
        E = np.zeros(num)
        C = np.zeros(num)
        D = np.zeros(num)
        init_state = copy.deepcopy(state)
        for a in range(num):
            # A[a], E[a] = self.get_reward(state, actions[a])
            C[a], _ = self.get_cost(state, actions[a])
            # D[a], _ = self.get_detla(state, actions[a])
            # max_n = actions[a] + self.num_actions
            new_state = self.state_transition(state, actions[a], E[a])
            # init_state = copy.deepcopy(new_state)
            max_n = actions[a] + self.num_actions - 1

            # if new_state.epoch < max_step:
            # A[a] = A[a] * factor + (1-factor) * self.discount_factor * self.value_function(new_state, max_step, max_n)
            # A[a] += self.discount_factor /Users/zber/ProgramDev/exp_pyTorch/results/grow_rank_cosine_20200626-170612/grow_rank_cosine__log.txt* self.value_function(new_state, max_step, max_n)

            A[a] = self.value_function(new_state, init_state, max_step, max_n, state)

            C[a] += self.discount_factor * self.cost_function(new_state, max_step, max_n)

            if A[a] < A[a - 1] and a > 0:
                break

            # D[a] += self.discount_factor * self.delta_function(new_state, max_step, max_n)
            # print(' V :{:.4f}'.format(A[a]))
            print('Action max list is {}'.format(new_state.list))
            print('\n')
        str_print = 'Action: {}, V: *{}&, DG:{}, C:{} ,loss:{}\n'.format(actions, A.tolist(), D.tolist(), C.tolist(),
                                                                         E.tolist())
        print(str_print, end='')
        write_log(str_print, self.log_path['path_to_log'])
        return A, E

    def state_transition(self, previous_state, action, estimate_loss):
        # action is the next N structure choosed
        new_state = copy.deepcopy(previous_state)
        new_state.n = action
        new_state.loss = estimate_loss
        new_state.epoch += 1
        new_state.w_shape[0] = action
        # new_state.next_in_shape[0] = action
        new_state.next_w_shape[1] = action
        new_state.list = []
        return new_state

    def get_reward(self, state, action, init_state=None):
        # get current reward
        # next_epoch = state.epoch + 1
        estimate_loss = self.obj_loss_mean.get_values(state.epoch + 1, action)
        # estimate_loss = self.obj_loss_mean.get_values(state.epoch, action)

        # delta loss discount
        # estimate_loss = estimate_loss + (0.30 ** (action - state.n))
        # estimate_loss = estimate_loss + estimate_loss * (1 - (0.9945 ** (action - state.n)))

        # estimate_loss = estimate_loss + estimate_loss * (1 - (0.9945 ** (action - state.n)))

        estimate_delta_loss = state.loss - estimate_loss
        # estimate_delta_loss = state.loss / estimate_loss

        # estimate_loss / cost function
        # grow_size = action - self.size
        delta_cost = self.obtain_delta_cost(state, action)

        # get init_state self decay a,b,c
        # a = self.obj_loss_mean.dic_self_decay[self.obj_loss_mean.seed_n[-1]][1][1]
        # base = self.obj_loss_mean.decay_n[(state.epoch+1)][1][1]

        # print('a: {}, b: {}'.format(a, b))
        # if init_state is not None:
        #     base = a ** (state.epoch + 1 - init_state.epoch)
        # else:
        #     base = a

        # diff_delta_loss = estimate_delta_loss - 1 if estimate_delta_loss - 1 > 0 else 0
        # diff_delta_loss = diff_delta_loss - diff_delta_loss * (1 - (base ** (action - init_state.n)))
        # estimate_delta_loss = 1 + diff_delta_loss

        # estimate_delta_loss = estimate_delta_loss - estimate_delta_loss * base * (1 - (b ** (action - init_state.n)))

        # estimate_delta_loss = estimate_delta_loss - estimate_delta_loss * (1 - (base ** (action - init_state.n)))

        estimate_value = estimate_delta_loss / delta_cost

        if estimate_value < 0:
            estimate_value = 0

        # state.loss = estimate_loss

        # print(
        #     'E {}, S {}-{} : g:{:.4f} , ng:{:.4f}, Dg:{:.4f} , Dc:{:.4f}, R:{:.4f}'.format(state.epoch+1, state.n, action, state.loss,
        #                                                                                    estimate_loss, estimate_delta_loss, delta_cost, estimate_value))
        # print('{}_{} : Dg:{},Dc:{},R:{}'.format(state, action, estimate_delta_loss, delta_cost, estimate_value))
        return estimate_value, estimate_loss

    def get_cost(self, state, action):
        estimate_loss = self.obj_loss_mean.get_values(state.epoch + 1, action)
        delta_cost = self.obtain_delta_cost(state, action)
        return delta_cost, estimate_loss

    def get_detla(self, state, action):
        estimate_loss = self.obj_loss_mean.get_values(state.epoch + 1, action)
        estimate_delta_loss = state.loss - estimate_loss
        # estimate_delta_loss = estimate_delta_loss / 100 - estimate_delta_loss / 100 * (1 - (0.99 ** (action - state.n)))
        return estimate_delta_loss, estimate_loss

    # @profile
    # grow V1
    # def grow(self):  # two extra parameters grow_function, model
    #     # is_grow = False
    #     # if len(self.array_estimate_loss) > 0:
    #     #     if self.train_loss[-1] > self.array_estimate_loss[-1]:
    #     #         return self.obj_loss_mean.seed_n[-1]
    #     self.obj_loss_mean.decay_n = {}
    #     actions = self.generate_actions(self.state)
    #     A, E = self.one_step_lookahead(self.state, actions)
    #     max_index = np.argmax(A)
    #     action = actions[max_index]
    #     estimate_loss = E[max_index]
    #     self.array_estimate_loss.append(estimate_loss)
    #     return action

    # v2
    def grow(self):  # two extra parameters grow_function, model
        # is_grow = False
        # if len(self.array_estimate_loss) > 0:
        #     if self.train_loss[-1] > self.array_estimate_loss[-1]:
        #         return self.obj_loss_mean.seed_n[-1]

        # control layer to layer selection
        # layer_switch = False

        self.obj_loss_mean.decay_n = {}
        actions = self.generate_actions(self.state)
        A, E = self.one_step_lookahead(self.state, actions)
        max_index = np.argmax(A)
        action = actions[max_index]
        estimate_loss = E[max_index]
        self.array_estimate_loss.append(estimate_loss)
        if action == actions[0] or len(self.input_shape) < 3 or layer_switch:
            res = [(action, A[max_index])]
            return res
        else:
            res = [(action, A[max_index]), (actions[0], A[0])]
            return res

    # update decay value
    def update(self, batch_loss, n, epoch):
        self.size = n
        self.epoch = epoch
        self.obj_loss_mean.put_values(batch_loss, epoch, n)
        mean = np.mean(batch_loss) * self.obj_loss_mean.c
        self.state = State(mean, n, epoch, self.input_shape, self.next_input_shape, self.weight_shape,
                           self.next_weight_shape)

    def update_loss(self, mean_loss):
        self.train_loss.append(mean_loss)

    def get_svrs(self):
        return self.obj_loss_mean.dic_self_decay

    def get_data(self):
        return self.obj_loss_mean.dic_data

    def update_cost(self, input_shape, weight_shape, next_input_shape, next_weight_shape):
        # update relative fields

        w_shape = []
        for i in range(len(weight_shape)):
            w_shape.append(weight_shape[i])
        self.weight_shape = w_shape

        w1_shape = []
        for i in range(len(next_weight_shape)):
            w1_shape.append(next_weight_shape[i])
        self.next_weight_shape = w1_shape

        self.input_shape = input_shape
        self.next_input_shape = next_input_shape

    def obtain_delta_cost(self, state, n):

        # return 1

        # current weight shape
        weight_shape = copy.deepcopy(state.w_shape)
        weight_shape[0] += (n - state.n)
        old_cur_cost = calculate_flops(state.in_shape, state.w_shape)
        cur_cost = calculate_flops(state.in_shape, weight_shape)

        # next weight shape
        next_weight_shape = copy.deepcopy(state.next_w_shape)
        next_weight_shape[1] += (n - state.n)
        old_next_cost = calculate_flops(state.next_in_shape, state.next_w_shape)
        next_cost = calculate_flops(state.next_in_shape, next_weight_shape)

        old_cost = old_cur_cost + old_next_cost
        new_cost = cur_cost + next_cost

        # memory size
        old_cur_size = obt_memory_size(state.w_shape)
        old_next_size = obt_memory_size(state.next_w_shape)

        new_cur_size = obt_memory_size(weight_shape)
        new_next_size = obt_memory_size(next_weight_shape)

        old_size = old_cur_size + old_next_size
        new_size = new_cur_size + new_next_size

        # delta_cost = new_cost / old_cost
        # delta_size = new_size / old_size
        #
        # total_delta = self.cost_factor * delta_cost + (1 - self.cost_factor) * delta_size

        cost1 = new_cost * self.cost_factor + (1 - self.cost_factor) * new_size
        cost2 = old_cost * self.cost_factor + (1 - self.cost_factor) * old_size

        c1 = new_cost * self.cost_factor * (1 - self.cost_factor) * new_size
        c2 = old_cost * self.cost_factor * (1 - self.cost_factor) * old_size

        total_delta1 = cost1 / cost2

        total_delta2 = c1 / c2

        if e_factor_switch:
            # total_delta = 1 + (total_delta1 - 1) * (math.e / 2) ** (n - self.size)
            # total_delta = 1 + (total_delta1 - 1) * (self.e_factor * math.e) ** (n - self.size)
            total_delta = 1 + (total_delta1 - 1) * self.e_factor * (math.e) ** (n - self.size)
        else:
            total_delta = 1 + (total_delta1 - 1) * self.e_factor

        # print('E {}, S from {} to {} : Cost1 {}, Cost2 {}, Cost3 {}'.format(state.epoch, state.n, n, total_delta1, total_delta2, total_delta))

        # total_delta = (delta_cost * delta_size)  # ** 1.2

        # cost decay
        if self.cost_decay:
            # step = (1 - self.decay_base) / self.cost_decay_step
            # alpha = self.decay_base + step * (self.epoch - num_base_epoch)
            # if alpha > 1:
            #     alpha = 1
            # total_delta = (total_delta - 1) * alpha + 1

            num_base_epoch = self.obj_loss_mean.num_essense
            step = 1 + (self.cost_base - self.cost_base ** (self.epoch - num_base_epoch + 1))
            total_delta = (total_delta - 1) * step + 1

        return total_delta


# class RIGrow1:
#     def __init__(self, model, train_loader, test_loader, criterion, optimizer, path, mode):
#         self.model = model
#         self.trian_loader = train_loader
#         self.test_loader = test_loader
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.path = path
#         ctl1 = GrowController(discount_factor=0.9, num_actions=10, threshold=0.09)
#         ctl2 = GrowController(discount_factor=0.9, num_actions=10, threshold=1.46)
#         ctl3 = GrowController(discount_factor=0.9, num_actions=10, threshold=1.46)
#         self.ctls = []
#         self.ctls.append(ctl1)
#         self.ctls.append(ctl2)
#         self.ctls.append(ctl3)
#         self.epoch = 0
#         self.ri_adaption = RIAdaption(self.model, self.optimizer, mode)
#         self.ri_gate = RIGate()
#
#     def grow_layers(self, num_grow):
#         self.model = grow_layers(self.model, num_grow)
#         self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#
#     def step(self, epoch):
#         self.epoch = epoch
#         print('the current model structure is {}'.format(self.model_size()))
#         self.train_one_epoch()
#         if epoch == 0:
#             # model needed to grow
#             self.grow_layers([2, 4, 5])
#         else:
#             num_grow = []
#             for i, controller in enumerate(self.ctls):
#                 action_picked = controller.grow()
#                 print('action {}, selected in layer {}'.format(action_picked, i))
#                 num_grow.append(action_picked)
#             num_grow = np.asarray(num_grow) - np.asarray(self.model_size())
#             self.grow_layers(num_grow)
#         print('After growing, the current model structre is {}'.format(self.model_size()))
#
#     def find_layers(self):
#         layers = []
#         for index_str, module in self.model.features._modules.items():
#             if isinstance(module, torch.nn.Conv2d) and int(index_str) == 0:
#                 layers.append(module)
#             elif isinstance(module, torch.nn.Conv2d) and int(index_str) > 0:
#                 layers.append(module)
#                 break
#
#         for index_str, module in self.model.classifier._modules.items():
#             if isinstance(module, torch.nn.Linear) and int(index_str) == 0:
#                 layers.append(module)
#                 break
#         return layers
#
#     def calculate_l2(self):
#         grads = []
#         for module in self.find_layers():
#             grad_tensor = module.weight.grad.data.cpu().numpy()
#             result = np.sqrt(np.sum(grad_tensor ** 2))
#             grads.append(result)
#         return grads
#
#     def percentage(self, list):
#         per_list = []
#         sum = np.sum(list)
#         for i in list:
#             per = i / sum
#             per_list.append(per)
#         return per_list
#
#     def model_size(self):
#         size = []
#         for module in self.find_layers():
#             size.append(module.bias.data.shape[0])
#         return size
#
#     def train_one_epoch(self):
#         train_loss, grads = train(self.trian_loader, self.model, self.criterion,
#                                   self.optimizer, self.path['path_to_log'],
#                                   epoch=self.epoch, two_output=True, print_grad=True)
#         # percentage may need to be more specific, size = batch loss
#         per_l2 = [self.percentage(grad) for grad in grads]
#         train_loss = np.reshape(train_loss, (1, -1))
#         per_l2 = np.transpose(per_l2, (1, 0))
#         per_loss_list = per_l2 * train_loss
#         # decay_self = self.generate_self_decay(length=100, loss=train_loss)
#         test(self.model, self.test_loader, self.criterion, self.path['path_to_log'], is_two_out=True)
#         for loss_list, n, controller in zip(per_loss_list, self.model_size(), self.ctls):
#             controller.update(loss_list, n, self.epoch)
#             controller.update_loss(np.mean(loss_list))
#         return np.mean(train_loss)

class RIGrow:
    # @profile
    def __init__(self, size, mode='LeNet', path=None):
        num_layers = size
        # df: 0.7 ->   , df: 0.75 -> saved_model
        if mode == 'MobileNet':
            costs = [0.8, 0.8, 0.8, 0.8]
            indexs = np.arange(0, size * 2, 2)
            # e_factors = [0.1, 0.1, 0.1, 0.1]
            # e_factors = [0.2, 0.2, 0.2, 0.2]
            # e_factors = [0.3, 0.3, 0.3, 0.3]
            e_factors = [0.4, 0.4, 0.4, 0.4]
        elif mode == 'VGG':
            costs = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
            indexs = np.arange(0, size, 1)
            # e_factors = [0.5, 0.4, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4]
            # e_factors = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
            # e_factors = [0.3] * 8
        else:
            costs = [0.8, 0.8, 0.2]
            indexs = np.arange(0, size, 1)

            # mnist
            # e_factors = [0.1, 0.1, 0.1]
            # e_factors = [0.2, 0.2, 0.2]
            e_factors = [0.3, 0.3, 0.3]
            # e_factors = [0.4, 0.4, 0.4]
            # e_factors = [0.6, 0.6, 0.6]
            # e_factors = [0.6, 0.45, 0.4]


            # sensing
            # e_factors = [1, 1, 1]
            # e_factors = [0.8, 0.8, 0.8]
            # e_factors = [0.6, 0.6, 0.6]

        base = 0.6
        f = 0.3
        self.num_essence = 3
        self.mode = mode
        self.ctls = [
            GrowController(discount_factor=0.5, num_actions=6, horizon_step=1, index=indexs[i], cost_factor=costs[i],
                           cost_base=base, e_factor=e_factors[i],
                           num_essence=self.num_essence, log_path=path) for i in range(num_layers)]

    def update_ctls(self, cos_sim, layer_size, epoch):
        for i, key in enumerate(cos_sim.keys()):
            self.ctls[i].update(cos_sim[key], layer_size[i], epoch)

    def update_cost(self, input_shapes, weight_shapes):
        if self.mode == 'MobileNet':
            for i in range(0, len(input_shapes) - 1, 2):

                if i == len(input_shapes) - 2:
                    self.ctls[i // 2].update_cost(input_shapes[i], weight_shapes[i], input_shapes[i + 1],
                                                  weight_shapes[i + 1])
                else:
                    self.ctls[i // 2].update_cost(input_shapes[i], weight_shapes[i], input_shapes[i + 2],
                                                  weight_shapes[i + 2])
        else:
            for i in range(len(input_shapes) - 1):
                self.ctls[i].update_cost(input_shapes[i], weight_shapes[i], input_shapes[i + 1], weight_shapes[i + 1])

        # for ctl, input_shape, weight_shape in zip(self.ctls, input_shapes, weight_shapes):
        #     ctl.update_cost(input_shape, weight_shape)

    # @profile
    # v1
    # def grow_ctls(self):
    #     growth_list = []
    #     for index, ctl in enumerate(self.ctls):
    #         action = ctl.grow()
    #         num_growth = action - ctl.size
    #         growth_list.append((ctl.index, num_growth))
    #
    #     return growth_list

    # v2
    def grow_ctls(self):
        growth_list = self._best_combine_options()

        # update e_factor in ctl
        self._update_e_factor(growth_list)
        size_list = [t[0] for t in growth_list]
        final_list = []
        for ctl, size in zip(self.ctls, size_list):
            num_growth = size - ctl.size
            final_list.append((ctl.index, num_growth))

        return final_list

    def _best_combine_options(self):
        import itertools

        growth_list = []
        init_channels = []
        fully_size = []

        for index, ctl in enumerate(self.ctls):
            actions = ctl.grow()
            init_channels.append(ctl.size)

            # actions = list(actions)
            # if is fully layer
            if len(ctl.input_shape) < 3:
                fully_size.append(actions[0])
                print("Linear layer option is {}".format(fully_size))
            else:
                growth_list.append(actions)

        # print option lists for each layer
        print(f"Layers Option List : {growth_list}")

        all_combs = list(itertools.product(*growth_list))

        best_index = 0
        best_score = 0
        input_shape = self.ctls[0].input_shape

        # init FLOPs
        init_flops = calculate_total_flops(model_name=self.mode, channels=init_channels, input_shape=input_shape)

        for index, comb in enumerate(all_combs):
            channels = []
            g_total = 0

            # calculate total_c and total_g
            for i_comb in comb:
                c = i_comb[0]
                g = i_comb[1]
                channels.append(c)
                g_total += g

            if fully_size:
                channels.append(fully_size[0][0])
                g_total += fully_size[0][1]

            total_flops = calculate_total_flops(model_name=self.mode, channels=channels, input_shape=input_shape)
            diff_flops = total_flops / init_flops

            if len(fully_size) == 0:
                f_size = len(channels)
            else:
                f_size = -len(fully_size)

            if [c for c in channels[:f_size]] != init_channels[:f_size]:
                score = g_total / diff_flops
            else:
                # if is not growing give a fix score
                score = -1000
            # score = g_total / total_flops

            if score > best_score:
                best_score = score
                best_index = index
            # each comb G score and Flops and score
            print("{} --> G: {}, Delta Flops:{}, Flops: {}, Score: {}".format([c[0] for c in comb], g_total, diff_flops,
                                                                              total_flops, score))

        # print best options
        print("Best --> {}".format(all_combs[best_index]))

        # add fully layers
        if fully_size:
            all_combs[best_index] = list(all_combs[best_index]) + [fully_size[0]]

        return all_combs[best_index]

    def update_df(self, epoch, base=0.4):
        df = 0.2 + base ** epoch
        print("==============  DF is {}".format(df))
        for ctl in self.ctls:
            ctl.discount_factor = df

    def _update_e_factor(self, best_list, lam=0.05):
        print("!!!!E-factor!!!!")
        for t, ctl in zip(best_list, self.ctls):
            if ctl.g_value is None:
                ctl.g_value = t[1]
            else:
                # update e_factor
                # ctl.e_factor = ctl.e_factor + lam * max(0, (ctl.g_value - t[1]))
                e = ctl.e_factor + lam * max(0, (ctl.g_value - t[1]))
                if e > 1:
                    e = 1
                ctl.e_factor = e
                print("New e_factor = {}".format(ctl.e_factor))
                ctl.g_value = t[1]

    def reset_e_factor(self, e_factor):
        for e, ctl in zip(e_factor, self.ctls):
            ctl.g_value = None
            ctl.e_factor = e


# class RIGrow:
#     # @profile
#     def __init__(self, model, path=None):
#         num_layers = len(find_modules_short(model))
#         # df: 0.7 ->   , df: 0.75 -> saved_model
#         costs = [0.8, 0.8, 0.2]
#         base = 0.3
#         f = 0.6
#         self.num_essence = 3
#         self.ctls = [
#             GrowController(discount_factor=0.5, num_actions=6, horizon_step=4, index=i, cost_factor=costs[i], cost_base=(base + f ** (i + 1)), num_essence=self.num_essence,
#                            log_path=path)
#             for i in range(num_layers)]
#
#     def update_ctls(self, cos_sim, layer_size, epoch):
#         for i, key in enumerate(cos_sim.keys()):
#             self.ctls[i].update(cos_sim[key], layer_size[i], epoch)
#
#     def update_cost(self, input_shapes, weight_shapes):
#         for i in range(len(input_shapes) - 1):
#             self.ctls[i].update_cost(input_shapes[i], weight_shapes[i], input_shapes[i + 1], weight_shapes[i + 1])
#
#         # for ctl, input_shape, weight_shape in zip(self.ctls, input_shapes, weight_shapes):
#         #     ctl.update_cost(input_shape, weight_shape)
#
#     # @profile
#     def grow_ctls(self):
#         growth_list = []
#         for index, ctl in enumerate(self.ctls):
#             action = ctl.grow()
#             num_growth = action - ctl.size
#             growth_list.append((index, num_growth))
#         return growth_list


if __name__ == '__main__':
    is_grow = True

    dataset = 'MNIST'  # CIFAR10,MNIST,EMG
    mode = 'ours'
    param = {
        'learning_rate': 0.0005,
        'epoch': 5,
        'batch_size': 100,
        'shuffle': True,
        'kernel_size': 5,
        'model_str': 'LeNet5_GROW'
    }

    model = LeNet5_GROW()
    path = dir_path('{}_{}_'.format(dataset, param['model_str']))
    trainloader, testloader = generate_data_loader(batch_size=param['batch_size'], dataset=dataset, is_main=False)

    # initialize critierion
    criterion = torch.nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'])

    grow_framework = RIGrow(model, trainloader, testloader, criterion, optimizer, path, mode)

    for i in range(param['epoch']):
        # trian
        grow_framework.step(i)
