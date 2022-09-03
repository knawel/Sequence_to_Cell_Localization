import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.theme import colors
import os


def plot_prg(log_folder):
    f_test = os.path.join(log_folder, "logprogress_test.txt")
    f_train = os.path.join(log_folder, "logprogress_train.txt")

    df_test = pd.read_csv(f_test, delimiter=';', header=None)
    df_train = pd.read_csv(f_train, delimiter=';', header=None)

    df_all = pd.DataFrame({'epoch': df_train[0], 'train_loss': df_train[1],
                           'test_loss': np.zeros(len(df_train[1]))})
    for e in df_test[0]:
        loss = df_test.loc[df_test[0] == e, 1].values[0]
        df_all.loc[df_all['epoch'] == e, 'test_loss'] = loss
    n_per_epoch = sum(df_all['epoch'] == 1)
    epochs = df_all.loc[:, 'epoch'].unique()

    fig, ax = plt.subplots()

    ax.plot(df_all.loc[:, 'train_loss'], '-.', label='train loss', c=colors[0])
    ax.plot(df_all.loc[:, 'test_loss'], '-.', label='test loss', c=colors[1])
    ax.set_xticks(np.arange(0, df_all.shape[0], n_per_epoch))
    ax.set_xticklabels(epochs)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    plt.savefig(os.path.join(log_folder, 'progress.png'))



