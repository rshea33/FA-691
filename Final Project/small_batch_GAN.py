from GAN import GAN
import pandas as pd


def main():
    data = pd.read_csv('Data/real_returns.csv').drop('Unnamed: 0', axis=1).T
    gan = GAN(data=data, epochs=10, batch_size=1)
    gan.train(epoch_verbose=True)
    samps = gan.sample(1000)
    print(samps.shape)
    samps = pd.DataFrame(samps.detatch().numpy())
    samps.to_csv('Data/small_batch.csv')

    gan.save_model('small_batch_GAN.pt')


if __name__ == '__main__':
    main()