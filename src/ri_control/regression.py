import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time
from sklearn.svm import LinearSVR


def plot_regression(svrs_dic, data, step=0.01):
    svrs = []
    kernel_label = []
    model_color = ['m', 'c', 'g', 'r', 'b']
    lw = 2

    for key in svrs_dic.keys():
        kernel_label.append(key)
        svrs.append(svrs_dic[key])

    num = len(svrs)

    fig, axes = plt.subplots(nrows=1, ncols=num, figsize=(num * 5, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        es = []
        losses = []
        for epoch_loss in data[kernel_label[ix]]:
            e, loss = epoch_loss
            es.append(e)
            losses += loss

        num = len(losses) / len(es)
        start = es[0] * num * step
        end = start + step * num * len(es)
        X = np.arange(start, end, step).reshape(-1, 1)
        y = losses
        end1 = start + 3 * step * num

        X1 = np.arange(start, end1, step).reshape(-1, 1)

        axes[ix].plot(X1, svr.predict(X1), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))
        # axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
        #                  edgecolor=model_color[ix], s=50,
        #                  label='{} support vectors'.format(kernel_label[ix]))
        # axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        #                  y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        #                  facecolor="none", edgecolor="k", s=50,
        #                  label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()


if __name__ == '__main__':

    # #############################################################################
    # Generate sample data
    # X = np.sort(5 * np.random.rand(40, 1), axis=0)
    # y = np.sin(X).ravel()

    y = [0.9919988678064512, 0.994226428760032, 0.9840103085887053, 0.9968272712202378, 0.9962250255736443, 0.990878152909484, 0.9956174796922254, 0.9819066908883789,
         0.9897905007596282, 0.9901563140262245, 0.9956193964673834, 0.9787637568051093, 0.9858574169648853, 0.9755819734822114, 0.965350511893007, 0.9731537373894752,
         0.9751162169266939, 0.9846727086387175, 0.9766689880725233, 0.964213016740543, 0.9812686107459627, 0.9808552488242802, 0.963019006521, 0.9918935016653305,
         0.9906510777041135,
         0.922220012838879, 0.9451180050592676, 0.9724466782983617, 0.9775114561377078, 0.9667283603528255, 0.9567076519477412, 0.9791640106626598, 0.9964588731756366,
         0.9746091357100052, 0.9783788991365207, 0.9740922144938768, 0.9906241808720823, 0.9578403764750918, 0.9811450868372724, 0.950190510817038, 0.9780203884419287,
         0.9485597984082081, 0.9671408982651895, 0.9773339140382311, 0.9666006449080722, 0.9678183398451834, 0.983623863563964, 0.983805671231138, 0.9705413852320529,
         0.9745084120890738, 0.9808037397698365, 0.9763554941226872, 0.9731349396831915, 0.9678303617431239, 0.9731306360514589, 0.9842910187153979, 0.9622349796679666,
         0.9815150481676771, 0.8769261808656581, 0.9818118730119495, 0.9429185926463609, 0.9343586041428187, 0.9956062706900949, 0.9814141520127905, 0.9792794493930578,
         0.9934629205900373, 0.9912266106947418, 0.9540987103335165, 0.9861111356878938, 0.9552788119351255, 0.9332891550313605, 0.9447930276082243, 0.9569581121328241,
         0.9502845020998465, 0.9915261624064788, 0.9623172176443571, 0.9599862339076429, 0.9775304760687167, 0.9665593315113495, 0.9612367903663904, 0.9883857788432426,
         0.9905057512594111, 0.9848951649060158, 0.9333503280192126, 0.9855449045699746, 0.9024877224012955, 0.9224194124572741, 0.9606303068576315, 0.9333137349185312,
         0.9318544686103812, 0.9532344438295193, 0.9456878347128077, 0.9571656701554548, 0.9738952133275304, 0.9350600979687285, 0.9840329848222896, 0.9835960752198797,
         0.8672228742199474, 0.9410851281262639, 0.9716996959713702]
    y = np.asarray(y) * 10
    X = np.arange(0, len(y)).reshape(-1, 1) / 100
    X1 = np.arange(0, len(y)).reshape(-1, 1) / 100

    # #############################################################################
    # Add noise to targets
    # y[::5] += 3 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=0.8, epsilon=.1,
                   coef0=1)
    svr_sigmoid = SVR(kernel='sigmoid', C=100, gamma='auto', degree=0.5, epsilon=.1,
                      coef0=1)

    svr_linear = LinearSVR(random_state=0, tol=1e-5)

    # #############################################################################
    # Look at the results
    lw = 2

    svrs = [svr_rbf, svr_lin, svr_poly]
    # svrs = [svr_linear]
    kernel_label = ['RBF', 'Linear', 'Polynomial']
    model_color = ['m', 'c', 'g', 'r', 'b']

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        print('Kernel is {}'.format(kernel_label[ix]))
        start = time.time()
        svr.fit(X, y)
        fit_end = time.time() - start
        print('fit time is {}'.format(fit_end))
        start_inf = time.time()
        yy = svr.predict(X)
        inf_end = time.time() - start_inf
        print('inference time is {}'.format(inf_end))
        print()

        axes[ix].plot(X1, svr.fit(X, y).predict(X1), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label='{} support vectors'.format(kernel_label[ix]))
        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()
