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
import numpy as np


def plot_im(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()

def load_model(file):
    model = SinkGen()
    model.load_state_dict(torch.load(file))

    return model

def generate_random_image(file):
    model = load_model(file)
    model.eval()
    z = Variable(torch.randn(1, model.latent_dim))
    print(z)
    x = model(z).view(-1,28,28)[0].data.numpy()
    #print(np.min(x), np.max(x))
    #x[x < .3] = 0
    #x[x > .3] = 1
    #print(x)
    plot_im(x)


class SinkGen(nn.Module):
    def __init__(self):
        super(SinkGen, self).__init__()

        self.image_dim = 28 # a 28x28 image corresponds to 4 on the FC layer, a 64x64 image corresponds to 13
                            # can calculate this using output_after_conv() in utils.py
        self.latent_dim = 100
        self.batch_size = 50

        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(.5),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(.5),
            nn.Linear(512, 28*28)#,
            #nn.Sigmoid(),#nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, z):
        x = self.generator(z)

        return x


def train(num_epochs = 100, batch_size = 128, learning_rate = 1e-4, reg = 4, eps = .01):
    train_dataset = dsets.MNIST(root='./data/',  #### testing that it works with MNIST data
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)


    sink_gen = SinkGen()
    sink_gen.batch_size = batch_size

    optimizer = torch.optim.RMSprop(sink_gen.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            #test_im = images[0].float().view(28,28).numpy()
            #plot_im(test_im)
            xr = Variable(images).view(-1, 28*28)
            xr += Variable(eps*torch.randn(xr.size()))

            z = Variable(torch.randn(sink_gen.batch_size, sink_gen.latent_dim))

            xg = sink_gen(z)

            M = pairwise_distances(xr, xg)

            loss = sink(M, reg=reg)
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

            torch.save(sink_gen.state_dict(), 'models/sinkhorn_generator/sink_gen_test_2.pkl')


if __name__ == '__main__':
    #train(batch_size = 128)
    model_file = 'models/sinkhorn_generator/sink_gen_test_2.pkl'
    #model = load_model(model_file)
    generate_random_image(model_file)

