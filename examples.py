from ot_pytorch import dmat, sink
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt

def uniform_example(batch_size = 100, reg = 10, filename = 'uniform_example1'):
    m_list = ((np.array(list(range(1, 100))) / 50.0 - 1)).tolist()
    loss = []
    for theta in m_list:
        x = np.zeros((batch_size, 2))
        y = np.zeros((batch_size, 2))
        x[:, 1] = np.random.uniform(0, 1, batch_size)
        y[:, 1] = np.random.uniform(0, 1, batch_size)
        y[:, 0] = theta

        x = Variable(torch.from_numpy(x).float())
        y = Variable(torch.from_numpy(y).float())

        M = dmat(x,y)
        loss.append(sink(M, reg=reg).data.numpy())


    plt.plot(m_list, loss)
    plt.xlabel('Theta')
    plt.ylabel('Sinkhorn Distance')
    plt.title('Uniform Example')
    fig_name = 'plots/uniform_example/' + filename + '.png'
    plt.savefig(fig_name)
    plt.show()

    df = pd.DataFrame({'theta': m_list, 'sink_dist': loss})
    data_name = 'data/uniform_example/' + filename + '.csv'
    df.to_csv(data_name)


def gaussian_example(batch_size = 100, reg = 10, dim = 10, filename = 'gaussian_example1'):
    m_list = range(21)
    loss = []
    for mu in m_list:
        x = np.random.normal(0, 1, (batch_size, dim))
        y = np.random.normal(mu, 1, (batch_size, dim))

        x = Variable(torch.from_numpy(x).float())
        y = Variable(torch.from_numpy(y).float())

        M = dmat(x, y)
        loss.append(sink(M, reg=reg).data.numpy())

    plt.plot(m_list, loss)
    plt.xlabel('Mu')
    plt.ylabel('Sinkhorn Distance')
    plt.title('Gaussian Example (Dim = ' + str(dim) + ')')
    fig_name = 'plots/gaussian_example/' + filename + '.png'
    plt.savefig(fig_name)
    plt.show()

    df = pd.DataFrame({'mu': m_list, 'sink_dist': loss})
    data_name = 'data/gaussian_example/' + filename + '.csv'
    df.to_csv(data_name)


if __name__ == '__main__':
    #uniform_example(filename='uniform_example2')
    gaussian_example(reg = 10000, dim = 700, filename='gaussian_example3')