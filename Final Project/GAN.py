import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from _param_dicts import *
import random
import numpy as np


class Discriminator(nn.Module):
    """
    Discriminator

    Overall architecture:
    - 5 inputs
    - 128 neurons (ReLU with dropout 0.2)
    - 1024 neurons (Tanh with dropout 0.4)
    - 256 neurons (ReLU)
    - 1 output (sigmoid)

    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 128), # 5 inputs
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1024),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1), # 1 output
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

class Generator(nn.Module):
    """
    Generator
    
    Overall architecture:
    - 5 inputs
    - 128 neurons (ReLU with dropout 0.2)
    - 256 neurons (Tanh with dropout 0.4)
    - 1024 neurons (ReLU)
    - 5 outputs

    """
    def __init__(self):


        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 128), # 5 inputs
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 5) # 5 outputs
        )

    def forward(self, x):
        return self.model(x)
    

class GAN():
    def __init__(
            self, 
            data,
            train_size=0.8,
            epochs=300,
            batch_size=128,
            lr=0.001,
            b1=0.9,
            b2=0.999,
            clip_value=None,
            random_state=None
            ):
        """
        Parameters:
        
        data : array-like, shape (n_samples, 6)
            data of log returns to train on
            final dimension is the label
            - 0 = fake
            - 1 = real

        train_size: : float, optional, default=0.8
            size of the training set

        epochs: : int, optional, default=1000
            number of epochs during training

        batch_size: : int, optional, default=128
            batch size for training

        lr: : float, optional, default=0.001
            learning rate for the optimizer

        b1: : float, optional, default=0.9
            beta 1 (for Adam optimizer)

        b2: : float, optional, default=0.999
            beta 2 (for Adam optimizer)

        clip_value: : float, optional, default=None
            clip value for the weights of the discriminator

        random_state: : int, optional, default=None
            random state for reproducibility

        Attributes:

        train_set : training set of length train_size * n_samples

        test_set : test set of length (1 - train_size) * n_samples

        device : device to use (GPU or CPU)

        generator : generator model

        discriminator : discriminator model

        adversarial_loss : loss function (binary cross entropy loss)

        optimizer_G : optimizer for the generator (Adam optimizer)

        optimizer_D : optimizer for the discriminator (Adam optimizer)


        Methods:

        train(self, verbose) : trains the model
            verbose : int, optional, default=False
                if 1, prints the loss at each epoch
                if 2, prints the loss at each batch

        sample(self, n_samples) : generates n_samples of fake data


        """

        if random_state is not None:
            torch.manual_seed(random_state)
    
        self.data = torch.tensor(data.values).float()

        self.train_size = train_size
        
        self.train_set, self.test_set = torch.utils.data.random_split(self.data,
            [
            int(len(self.data) * self.train_size),
            len(self.data) - int(len(self.data) * self.train_size)
            ]
        )
        
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.lr = lr 
        self.b1 = b1 
        self.b2 = b2 
        
        self.clip_value = clip_value 
        
        
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ) # if GPU is available, use it

        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.adversarial_loss = nn.BCELoss() # binary cross entropy loss

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2)
        )

        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2)
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True
        )



    def train(self, verbose=0):
        for epoch in range(self.epochs):
            for n, real_samples in enumerate(self.train_loader):

                # data for training the discriminator
                real_samples_labels = torch.ones((self.batch_size, 1))
                latent_space_samples = torch.randn((self.batch_size, 5))
                gen_samples = self.generator(latent_space_samples)
                gen_sample_labels = torch.zeros((self.batch_size, 1))

                # make sure the real_sampeles and gen_samples are the same size
                if len(real_samples) != len(gen_samples):
                    r_len, g_len = len(real_samples), len(gen_samples)
                    length = min(r_len, g_len)
                    real_samples = real_samples[:length]
                    gen_samples = gen_samples[:length]
                    real_samples_labels = real_samples_labels[:length]
                    gen_sample_labels = gen_sample_labels[:length]

                all_samples = torch.cat((real_samples, gen_samples))
                all_sample_labels = torch.cat((
                    real_samples_labels,
                    gen_sample_labels
                ))


                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.discriminator.zero_grad()
                self.optimizer_D.zero_grad()

                output_D = self.discriminator(all_samples)

                loss_D = self.adversarial_loss(output_D, all_sample_labels)
                loss_D.backward()

                self.optimizer_D.step()

                # Clip weights of discriminator
                if self.clip_value is not None:
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                # -----------------

                # data for training the generator
                
                if self.batch_size != len(real_samples):
                    latent_space_samples = torch.randn((len(real_samples), 5))
                else:
                    latent_space_samples = torch.randn((self.batch_size, 5))

                # -----------------
                #  Train Generator
                # -----------------

                self.generator.zero_grad()
                self.optimizer_G.zero_grad()

                generated_samples = self.generator(latent_space_samples)
                output_D_generated = self.discriminator(generated_samples)

                loss_G = self.adversarial_loss(output_D_generated, real_samples_labels)

                loss_G.backward()
                self.optimizer_G.step()

                if verbose == 2:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch+1, self.epochs, n, len(self.train_loader), loss_D.item(), loss_G.item())
                    )
                
            if verbose == 1:
                print(
                    "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, self.epochs, loss_D.item(), loss_G.item())
                )



    def sample(self, n_samples, plot=False):
        """
        Generates n_samples of fake data
        """
        noise = torch.randn(n_samples, 5).to(self.device)
        if plot:
            plt.plot(self.generator(noise).cpu().detach().numpy().T)
            plt.show()

        return self.generator(noise)
    

    def save_model(self, path):
        """
        Save the model as a 'pt` file
        """
        if not path.endswith('.pt'):
            path += '.pt'

        torch.save(self.generator.state_dict(), path)


def main():
    
    data = pd.read_csv('Data/real_returns.csv').drop('Unnamed: 0', axis=1).T

    d = {
        'data': [data],
        'train_size': [0.7, 0.8, 0.9, 0.95],
        'epochs': [25, 50, 100, 200, 250, 300, 400, 500],
        'batch_size': [32, 64, 128, 256, 512, 1024],
        'lr': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'b1': [0.5, 0.6, 0.7, 0.8, 0.9],
        'b2': [0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999],
        'clip_value': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, None],
        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }


    for i in range(100):
        params = {}
        for k in d.keys():
            params[k] = random.choice(d[k])

        print(i+1)
        gan = GAN(**params)
        gan.train(full_verbose=True)
        samps = gan.sample(1000)
        samps = pd.DataFrame(samps.detach().numpy())
        samps.to_csv(f'Data/fake_returns_GAN_{i+1}.csv')
        gan.save_model(f'model_{i+1}.pt')

    
    # print(gan.data.shape)
    # print(len(gan.train_set), len(gan.test_set[1]))
    # print(len(gan.test_set), len(gan.test_set[1]))
    # print(gan.discriminator)
    # print(gan.generator)
    gan.train(full_verbose=True)

    fake_data = gan.sample(100, plot=True)

    fake_data = pd.DataFrame(fake_data.detach().numpy())
    print(fake_data.shape)
    print(fake_data)
    fake_data.to_csv('Data/fake_returns_GAN.csv')

    gan.save_model('model.pt')

if __name__ == '__main__':
    main()