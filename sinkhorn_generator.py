import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from ot_pytorch import sink, pairwise_distances, dmat
import pandas as pd
from matplotlib import pyplot as plt
import time


def plot_im(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()


class SinkGen(nn.Module):
    def __init__(self):
        super(SinkGen, self).__init__()

        self.image_dim = 28 # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 20
        self.batch_size = 50

        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 16*16),
            nn.ReLU(),
            nn.Linear(16*16, 28*28),
            nn.Tanh()
        )


    def forward(self, z):
        x = self.generator(z)

        return x


def train(num_epochs = 100, batch_size = 128, learning_rate = 1e-4):
    train_dataset = dsets.MNIST(root='./data/',  #### testing that it works with MNIST data
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    #bvae = AE()
    #bvae.batch_size = batch_size

    sink_gen = SinkGen()
    sink_gen.batch_size = batch_size

    optimizer = torch.optim.Adam(sink_gen.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            xr = Variable(images).view(-1, 28*28)
            #print(xr.size())

            z = Variable(torch.rand(sink_gen.batch_size, sink_gen.latent_dim))

            # Forward + Backward + Optimize
            xg = sink_gen(z)
            #test = xg[0].view(28,28).data.numpy()
            #test[test >= .5] = 1
            #test[test < .5] = 0
            #plot_im(test)
            #print(test)
            #print(xg[0])
            #print(xg.size())

            M = dmat(xr,xg)#pairwise_distances(xr, xg)
            #print(M)

            loss = sink(M, reg=10)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

        torch.save(sink_gen.state_dict(), 'sink_gen_test.pkl')

if __name__ == '__main__':
    train()
