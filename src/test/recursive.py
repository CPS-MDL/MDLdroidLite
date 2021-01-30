import numpy as np


class State:
    def __init__(self, loss, n, epoch):
        self.loss = loss
        self.n = n
        self.epoch = epoch

    def __str__(self):
        return 'size: {},loss: {},epoch: {}'.format(self.n, self.loss, self.epoch)


def tf(loss, n, epoch):
    return State(loss, n, epoch)


def recur_fn(state, max_epoch, max_n, df=0.9):  #
    if state.epoch >= max_epoch:
        # actions = []
        # for a in range(state.n, max_n):
        #     actions.append(a)
        # print(actions)
        # return np.mean(actions)
        return state.n
    else:
        actions = []
        for a in range(state.n, max_n):
            new_state = tf(0, a, state.epoch + 1)
            actions.append(recur_fn(new_state, max_epoch, max_n))
        print(actions)
        return state.n + df * np.mean(actions)


if __name__ == "__main__":
    m_max = 5
    n_max = 6
    start = 0
    epoch = 0
    max_epoch = start + m_max
    max_n = start + n_max
    state = State(0, 1, 1)
    b = recur_fn(state, m_max, n_max)
    print(b)