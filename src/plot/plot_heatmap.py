import numpy as np;

np.random.seed(0)
import seaborn as sns;

sns.set()


def heatmap(ndarry):
    ndarry = np.abs(ndarry)
    ax = sns.heatmap(ndarry)
