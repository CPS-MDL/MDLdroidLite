# backup of grow controller
class GrowController:
    def __init__(self, decay, discount_factor, num_actions, threshold):  # , init_model, trian, test
        self.decay = decay
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.threshold = threshold
        # self.model = init_model
        self.std = []
        self.mini = []
        self.state = None
        self.num_neuron = []
        self.loss = []
        self.distribution_delta_loss = None
        self.distribution_delta_std = None
        self.same_decay = None
        self.epoch = 1
        self.delta_loss = None
        self.delta_mean = None
        self.train_loss = None
        self.dic_self_decay = {}
        self.dic_self_delta_loss = {}
        self.dic_decay = {}
        self.loss_mean = DynamicMemory(20, 100)

    def half_normal(self, min_value, std, expected_loss):
        # pro = halfnorm.pdf((expected_loss * -1), loc=(min_value * -1), scale=std)
        # pro = norm(min_value, std).pdf(expected_loss)
        pro = 1 - halfnorm.cdf((expected_loss * -1), loc=(min_value * -1), scale=std)
        return pro

    def generate_actions(self, state):
        """
        this action set includes the current structure N
        :param state: state passed
        :return:  return an array of actions for the state passed
        """
        current_n = state.n
        actions = [i for i in range(current_n, current_n + self.num_actions)]
        return actions

    def get_decay(self, action):
        return self.same_decay if action == self.state.n else self.decay

    def value_function(self, state, delta_loss, discount_factor):
        # state is current statec
        # action is current action
        # action_size
        if delta_loss < self.threshold:
            return 0
        else:
            actions = self.generate_actions(state)
            decayed_mins = [self.calculate_decay(state, action, self.mini[-1], self.distribution_delta_loss) for action
                            in actions]
            decayed_std = [self.calculate_decay(state, action, self.std[-1], self.distribution_delta_std) for action in
                           actions]
            # expected_loss = state.loss - delta_loss
            expected_loss = [self.calculate_decay(state, action, self.mean[-1], self.distribution_delta_std) for action
                             in
                             actions]
            pro_for_actions = [self.half_normal(min_value, std, expected_loss) for min_value, std, expected_loss in
                               zip(decayed_mins, decayed_std, expected_loss)]
            pro_for_n = np.asarray(pro_for_actions) / np.asarray(actions)
            max_index = np.argmax(pro_for_n)
            value = pro_for_n[max_index]
            selected_action = actions[max_index]

            state = self.state_transition(state, selected_action, delta_loss)
            print(state)
            delta_loss = self.decay * delta_loss
            return value + discount_factor * self.value_function(state, delta_loss, discount_factor)

    def get_expected_loss(self, old_state, new_state):
        expected_loss = new_state.loss - (old_state.loss - new_state.loss) * self.decay
        return expected_loss

    def one_step_lookahead(self, state, actions, delta_loss):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            A[a] = self.get_reward(state, actions[a], delta_loss)
            new_state = self.state_transition(state, actions[a], delta_loss)
            A[a] += self.discount_factor * self.value_function(new_state, delta_loss, self.discount_factor)
        return A

    def calculate_decay(self, state, action, val, delta_decay):
        if state.n == action:
            decay = self.same_decay
        else:
            decay = self.decay
        diff = action - state.n + 1
        for i in range(1, diff):
            val -= delta_decay ** decay
        return val

    def state_transition(self, previous_state, action, delta_loss):
        # action is the next N structure choosed
        new_state = copy.copy(previous_state)
        if action == new_state.n:
            new_state.iteration += 1
        else:
            new_state.epoch += 1
        new_state.n = action
        new_state.loss = new_state.loss - delta_loss * self.decay ** (action - previous_state.n)
        # update state loss = old_loss - delta_loss * decay
        return new_state

    def get_reward(self, state, action, delta_loss):
        decayed_mins = self.calculate_decay(state, action, self.mini[-1], self.distribution_delta_loss)
        decayed_std = self.calculate_decay(state, action, self.std[-1], self.distribution_delta_std)
        expected_loss = self.calculate_decay(state, action, self.loss[-1], self.delta_loss)
        pro_for_actions = self.half_normal(decayed_mins, decayed_std, expected_loss)
        pro_for_n = pro_for_actions / action
        return pro_for_n

    def set_train_loss(self, train_loss):
        self.train_loss = train_loss

    def append_values(self, std, mean, mini, loss, num_neuron):
        self.mean.append(mean)
        self.std.append(std)
        self.loss.append(loss)
        self.mini.append(mini)
        self.num_neuron.append(num_neuron)

    def grow(self):  # two extra parameters grow_function, model
        actions = self.generate_actions(self.state)
        delta_loss = abs(self.mean[-1] - self.mean[-2])
        delta_std = abs(self.std[-1] - self.std[-2])
        self.threshold = 0.1 * delta_loss
        actions_with_value = self.one_step_lookahead(self.state, actions, delta_loss)
        action = np.max(actions_with_value)
        expected_loss = self.state.loss - delta_loss
        # run the grow function
        # get the new loss
        # update decay value
        # compare with next epoch of current N
        # 1. generalise actions
        # 2. get new expected loss according to AL
        #    get threshold 5% of AL
        # 3. for each actions, generate their PDF based on the decayed mean and decayed std
        # 4. calculate their expected value based on PDF and time
        #  e.g. expected_value = current value + next value
        #  A[a] += prob * (reward + discount_factor * V[next_state])
        # prob = next_loss on each pdf
        # reward = the value produced by this action
        # V[next_state] = delta_next_loss /  n
        #  update a new
        # 5. choose the action with the maximum values
        # 6. train and get real loss, mean ,std
        # 7. update the Value function
        # e.g new_loss - last_loss / AN
        # update_decay
        return action, expected_loss

    # update decay value
    def update(self, same_decay, is_grow=True):
        if is_grow:
            self.distribution_delta_loss = abs(
                (self.mean[-1] - self.mean[-2]) / (self.num_neuron[-1] - self.num_neuron[-2]))
            self.distribution_delta_std = abs(
                (self.std[-1] - self.std[-2]) / (self.num_neuron[-1] - self.num_neuron[-2]))
            if len(self.mean) > 2:
                self.decay = (self.mean[-1] - self.mean[-2] / (self.num_neuron[-1] - self.num_neuron[-2])) / (
                        self.mean[-2] - self.mean[-3] / (self.num_neuron[-2] - self.num_neuron[-3]))
            self.same_decay = same_decay
        else:
            self.same_decay = same_decay
        # update state
        self.epoch = self.epoch + 1
        self.state = State(loss=self.loss[-1], n=self.num_neuron[-1], iteration=1, epoch=self.epoch)
        self.delta_loss = (self.mean[-2] - self.mean[-1])  # only change when it grows
        self.threshold = self.delta_loss * 0.1



# import numpy as np
# from scipy.stats import halfnorm
# import copy
# import math
#
#
#
# def nth_root(num, root):
#     # ** means square
#     answer = num ** (1 / root)
#     return answer
#
#
# # State  - the features of the state, previous state
# # {epoch: ,N : , epoch_for_N , loss:}
#
#
# # Transition - State transition method
# # Action - Action linear
# # Reward - Value function - how good * is to be in a state v(s) to take an action from that state q(s,a)
# # Policy pai(a|s) The prediction of future reward taken from this environment.
# # discount factor - d
# # Threshold T
#
# # param
#
#
# # mean = lots of sampled loss, calculate the mean
# # std = lots of sampled loss, calculate the std
#
# # 1. how to define state
# # 2. Find a linear function to match the relationship between loss and n
# # 3. time remain same but loss decay
#
#
# # MDP The standard framework for modeling sequential decision making or planning under uncertainty
# """
# __Args__:
#     1. policy: [S, A] shaped matrix representing the policy.
#     2. env: OpenAI env.
#         i.  env.P transition probabilities of the environment.
#         ii. env.P[s][a] is a list of transition tuples P[s][a] == [(probability, nextstate, reward, done), ...].
#         iii.env.nS is a number of states in the environment.
#         iv. env.nA is a number of actions in the environment.
#     3. discount_factor: Gamma discount factor.
#     4. theta: We stop evaluation once our value function change is less than theta for all states.
# __Returns__:
#     Vector of length env.nS representing the value function.
#     Matrix of length env.nS x env.nA representing the policy.
#
# MDP env:
#     - env.nS = 16
#       s ∈ S = {0...15}, where 0 and 15 are terminal states
#     - env.nA = 4,
#       a ∈ A = {UP = 0,RIGHT = 1,DOWN = 2,LEFT = 3}
#     - P[s][a]= {P[s][UP],P[s][RIGHT], P[s][DOWN], P[s][LEFT]}  : state transition function specifying P(ns_up|s,UP).
#       i.e. P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))] # Not a terminal state
#            P[s][UP] = [(1.0, s, reward, True)] # A terminal state
#     - R is a reward function R(s,a,s')
#       reward = 0.0 if we are stuck in a terminal state, else -1.0
#
# Local Variables
# - policy[s]: action array
# - chosen_a: a real number ∈ {0,1,2,3}
# - action_values: action values array
# - best_a: integer variable
# - policy_stable: boolean variable
# - delta: integer variable
# - V[S]: real array
# """
#
#
# def half_normal(min_value, std, expected_loss):
#     pro = halfnorm.pdf(expected_loss, loc=min_value, scale=std)
#     return pro
#
#
# def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
#     # Initialize thel value function
#     V = np.zeros(env.nS)
#     # While our value function is worse than the threshold theta
#     while True:
#         # Keep track of the update done in value function
#         delta = 0
#         # For each state, look ahead one step at each possible action and next state
#         for s in range(env.nS):
#             v = 0
#             # The possible next actions, policy[s]:[a,action_prob]
#             for a, action_prob in enumerate(policy[s]):
#                 # For each action, look at the possible next states,
#                 for prob, next_state, reward, done in env.P[s][
#                     a]:  # state transition  P[s][a] == [(prob, nextstate, reward, done), ...]
#                     # Calculate the expected value function
#                     v += action_prob * prob * (
#                             reward + discount_factor * V[next_state])  # P[s, a, s']*(R(s,a,s')+γV[s'])
#                     # How much our value function changed across any states .
#             delta = max(delta, np.abs(v - V[s]))
#             V[s] = v
#         # Stop evaluating once our value function update is below a threshold
#         if delta < theta:
#             break
#     return np.array(V)
#
#
# def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
#     # Initiallize a policy arbitarily
#     policy = np.ones([env.nS, env.nA]) / env.nA
#
#     while True:
#         # Compute the Value Function for the current policy
#         V = policy_eval_fn(policy, env, discount_factor)
#
#         # Will be set to false if we update the policy
#         policy_stable = True
#
#         # Improve the policy at each state
#         for s in range(env.nS):
#             # The best action we would take under the currect policy
#             chosen_a = np.argmax(policy[s])
#             # Find the best action by one-step lookahead
#             action_values = np.zeros(env.nA)
#             for a in range(env.nA):
#                 for prob, next_state, reward, done in env.P[s][a]:
#                     action_values[a] += prob * (reward + discount_factor * V[next_state])
#             best_a = np.argmax(action_values)
#
#             # Greedily (max in the above line) update the policy
#             if chosen_a != best_a:
#                 policy_stable = False
#             policy[s] = np.eye(env.nA)[best_a]
#
#         # Until we've found an optimal policy. Return it
#         if policy_stable:
#             return policy, V
#
#
# decay
#
#
# # temp_decay = array with size state.
#
#
# # this funciton used to get the reward value, which can be recursively applied
# def value_function(state, next_state, decay):
#     # when loss is fixed, we need to compare the time
#     # unpack the state
#
#     old_loss, old_n = state
#     new_loss, new_n = next_state
#
#     previous_n = state['N']
#     v = loss / action - previous_n
#     return 1
#
#
# def value_function(state, expected_loss, discount_factor):
#     # state is current statec
#     # action is current action
#     # action_size
#     if state.loss - expected_loss < Threshold:
#         return 0
#     else:
#         actions = generate_action(state)
#         decayed_means = [mean[-1] * decay ** (i - state.n) for i in actions]
#         decayed_std = [std[-1] * decay ** (i - state.n) for i in actions]
#         pro_for_actions = [half_normal(min_value, std, expected_loss) for min_value, std in
#                            zip(decayed_means, decayed_std)]
#         pro_for_n = np.asarray(pro_for_actions) / np.asarray(actions)
#         max_index = np.argmax(pro_for_n)
#         value = pro_for_n[max_index]
#         selected_action = actions[max_index]
#         new_state = state_transition(state, selected_action)
#         new_expected_loss = new_state.loss - decay * (state.loss - new_state.loss)
#
#     return value + discount_factor * value_function(new_state, new_expected_loss, discount_factor)
#
#
# def value_function_current(state, next_state, decay):
#     # when loss is fixed, we need to compare the time
#     # unpack the state
#
#     old_loss, old_n = state
#     new_loss, new_n = next_state
#
#     previous_n = state['N']
#     v = loss / action - previous_n
#     return 1
#
#
# # experiment
# l_0 = 1.8
# N_0 = 4
# l_1 = 1.6
# N_1 = 8
# decay = (l_0 - l_1) / (N_0 - N_1)
# action_size = 10
#
#
# def state_transition(previous_state, action):
#     # action is the next N structure choosed
#     new_state = copy.copy(previous_state)
#     if action == new_state.n:
#         new_state.iteration += 1
#     else:
#         new_state.epoch += 1
#     new_state.n = action
#     new_state.loss = new_state.loss * decay
#     return new_state
#
#
# # discount  = 0.8
# # nA = 10
#
#
# def one_step_lookahead(state, actions, expected_loss):
#     """
#     Helper function to calculate the value for all action in a given state.
#
#     Args:
#         state: The state to consider (int)
#         V: The value to use as an estimator, Vector of length env.nS
#
#     Returns:
#         A vector of length env.nA containing the expected value of each action.
#     """
#     A = np.zeros(num_actions)
#     for a in range(num_actions):
#         A[a] = get_reward(state, actions[a], expected_loss)
#         new_state = state_transition(state, actions[a])
#         expected_loss = new_state.loss - (state.loss - new_state.loss) * decay
#         A[a] += discount_factor * value_function(state, expected_loss, discount_factor)
#     return A
#
#
# class MDP_grow:
#     def __init__(self, decay, discount_factor, num_actions, threshold, init_model, trian, test):
#         self.decay = decay
#         self.discount_factor = discount_factor
#         self.num_actions = num_actions
#         self.threshold = threshold
#         self.model = init_model
#         self.mean = []
#         self.std = []
#         self.mini = []
#         self.state = None
#
#     def generate_actions(self, state):
#         """
#         :param state: state passed
#         :return:  return an array of actions for the state passed
#         """
#         current_n = state.n
#         actions = [for i in range(current_n, current_n + self.num_actions)]
#         return actions
#
#     def value_function(self, state, expected_loss, discount_factor):
#         # state is current statec
#         # action is current action
#         # action_size
#         if state.loss - expected_loss < self.threshold:
#             return 0
#         else:
#             actions = self.generate_action(state)
#             decayed_means = [self.mean[-1] * decay ** (i - self.state.n) for i in actions]
#             decayed_std = [self.std[-1] * decay ** (i - self.state.n) for i in actions]
#             pro_for_actions = [half_normal(min_value, std, expected_loss) for min_value, std in
#                                zip(decayed_means, decayed_std)]
#             pro_for_n = np.asarray(pro_for_actions) / np.asarray(actions)
#             max_index = np.argmax(pro_for_n)
#             value = pro_for_n[max_index]
#             selected_action = actions[max_index]
#             new_state = state_transition(state, selected_action)
#             new_expected_loss = new_state.loss - decay * (state.loss - new_state.loss)
#
#         return value + discount_factor * self.value_function(new_state, new_expected_loss, discount_factor)
#
#     def get_expected_loss(self, old_state, new_state):
#         expected_loss = new_state.loss - (old_state.loss - new_state.loss) * self.decay
#         return expected_loss
#
#     def one_step_lookahead(self, state, actions, expected_loss):
#         """
#         Helper function to calculate the value for all action in a given state.
#
#         Args:
#             state: The state to consider (int)
#             V: The value to use as an estimator, Vector of length env.nS
#
#         Returns:
#             A vector of length env.nA containing the expected value of each action.
#         """
#         A = np.zeros(self.num_actions)
#         for a in range(self.num_actions):
#             A[a] = self.get_reward(state, actions[a], expected_loss)
#             new_state = state_transition(state, actions[a])
#             expected_loss = new_state.loss - (state.loss - new_state.loss) * decay
#             A[a] += self.discount_factor * self.value_function(state, expected_loss, self.discount_factor)
#         return A
#
#     def state_transition(self, previous_state, action):
#         # action is the next N structure choosed
#         new_state = copy.copy(previous_state)
#         if action == new_state.n:
#             new_state.iteration += 1
#         else:
#             new_state.epoch += 1
#         new_state.n = action
#         new_state.loss = new_state.loss * self.decay
#         return new_state
#
#     def get_reward(self, state, action, expected_loss):
#         decayed_means = self.mean[-1] * decay ** (action - state.n)
#         decayed_std = self.std[-1] * decay ** (action - state.n)
#         pro_for_actions = half_normal(decayed_means, decayed_std, expected_loss)
#         pro_for_n = pro_for_actions / action
#         return pro_for_n
#
#     def run(self):
#         i = 1
#         while True:
#             # init
#             if i == 1:
#                 self.mean.append(1)
#                 self.std.append(1)
#                 self.loss.append(1)
#                 self.mini.append(1)
#             # gusses
#             elif i == 2:
#
#                 self.mean.append(2)
#                 self.std.append(2)
#                 self.loss.append(2)
#                 self.mini.append(2)
#                 self.decay = 0
#             else:
#                 state = State(1, 3, 2)
#                 actions = self.generate_action(state)
#                 delta_loss = self.mean[-1] - self.mean[-2]
#                 self.threshold = 0.05 * delta_loss
#                 expected_loss = delta_loss * self.decay
#                 actions_with_value = self.one_step_lookahead(state, actions, expected_loss)
#                 action = np.max(actions_with_value)
#                 # run the grow function
#                 # get the new loss
#                 # update decay value
#                 # compare with next epoch of current N
#                 # 1. generalise actions
#                 # 2. get new expected loss according to AL
#                 #    get threshold 5% of AL
#                 # 3. for each actions, generate their PDF based on the decayed mean and decayed std
#                 # 4. calculate their expected value based on PDF and time
#                 #  e.g. expected_value = current value + next value
#                 #  A[a] += prob * (reward + discount_factor * V[next_state])
#                 # prob = next_loss on each pdf
#                 # reward = the value produced by this action
#                 # V[next_state] = delta_next_loss /  n
#                 #  update a new
#                 # 5. choose the action with the maximum values
#                 # 6. train and get real loss, mean ,std
#                 # 7. update the Value function
#                 # e.g new_loss - last_loss / AN
#                 # update_decay
#                 if 0 < self.threshold:
#                     break
#                 i += 1
#
#
# class State:
#     def __init__(self, loss, n, iteration, epoch):
#         self.loss = loss
#         self.n = n
#         self.iteration = iteration
#         self.epoch = epoch
#         self.is_done = False
#
#
# def main():
#     i = 1
#     mean = []
#     std = []
#     mini = []
#     loss = []
#     delta = 0
#     decay = 0.9
#     threshold = 1
#     action_size = 10
#     N_sequences = []
#
#
# def value_iteration(env, theta=0.0001, discount_factor=1.0):
#     # Look ahead one step at each possible action and next state (full backup)
#     def one_step_lookahead(state, action, V):
#         """
#         Helper function to calculate the value for all action in a given state.
#
#         Args:
#             state: The state to consider (int)
#             V: The value to use as an estimator, Vector of length env.nS
#
#         Returns:
#             A vector of length env.nA containing the expected value of each action.
#         """
#         A = np.zeros(env.nA)
#         for a in range(env.nA):
#             for prob, next_state, reward, done in env.P[state][a]:
#                 A[a] += prob * (reward + discount_factor * V[next_state])
#         return A
#
#     V = np.zeros(env.nS)
#     while True:
#         # Stopping condition
#         delta = 0
#         # Update each state...
#         for s in range(env.nS):
#             # Do a one-step lookahead to find the best action
#             A = one_step_lookahead(s, V)
#             best_action_value = np.max(A)
#             # Calculate delta across all states seen so far
#             delta = max(delta, np.abs(best_action_value - V[s]))
#             # Update the value function
#             V[s] = best_action_value
#             # Check if we can stop
#         if delta < theta:
#             break
#
#     # Create a deterministic policy using the optimal value function
#     policy = np.zeros([env.nS, env.nA])
#     for s in range(env.nS):
#         # One step lookahead to find the best action for this state
#         A = one_step_lookahead(s, V)
#         best_action = np.argmax(A)
#         # Always take the best action
#         policy[s, best_action] = 1.0
#
#     return policy, V
#
