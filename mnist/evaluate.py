import pickle

import numpy as np

import train
from model_parameters import *
from matplotlib import pyplot as plt


def plot_style_embeddings(enc, dec):
    data, _ = train.get_data(NUM_BATCH)
    # Get all the embeddings
    Zs = np.zeros((len(data), NUM_BATCH, NUM_STYLE))
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == NUM_BATCH:
            images = images.view(-1, NUM_PIXELS)
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=1)
            z = q['z'].value.cpu().detach().squeeze().numpy()
            Zs[b] = z
    Zs = Zs.reshape(-1, NUM_STYLE)

    figs, axes = plt.subplots(NUM_STYLE, NUM_STYLE, figsize=(30, 30), sharex=True)
    figs.suptitle(r'$Z (style) \ Embeddings$', fontsize=30)

    for i in range(NUM_STYLE):
        axes[NUM_STYLE - 1, i].set_xlabel(r'$\mathbf{z_{%d}}$' % i, fontsize=10)
        axes[i, 0].set_ylabel(r'$\mathbf{z_{%d}}$' % i, fontsize=10)
        for j in range(NUM_STYLE):
            ax = axes[j, i]
            ax.set(adjustable='box', aspect='equal')
            ax.set_xticks(np.linspace(-2, 2, num=5))
            if i == j:
                ax.tick_params(
                    axis='y',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    left=False,  # ticks along the bottom edge are off
                    right=False,  # ticks along the top edge are off
                    )
                ax.hist(Zs[:, i], bins=40)
            else:
                ax.set_yticks(np.linspace(-2, 2, num=5))
                ax.scatter(Zs[:, i], Zs[:, j], alpha=0.5)
    plt.savefig("output/embeddings.png",bbox_inches='tight')
    plt.clf()


def plot_traversals(enc, dec):
    def vary_z2(index, zmin, zmax):
        f, axarr = plt.subplots(NUM_STYLE, NUM_DIGITS, figsize=(10, 10), sharey=True)
        f.suptitle(r'$\mathbf{z_{%d}} \ \  varying$' % index, fontsize=30)
        z_range = np.linspace(zmin, zmax, num=NUM_STYLE)

        for i in range(NUM_STYLE):
            for j in range(NUM_DIGITS):
                null_image = torch.zeros((1, 784))
                z = torch.zeros((1, NUM_STYLE))
                y_hot = torch.zeros((1, 10))
                z[0, index] = z_range[i]
                y_hot[0, j] = 1
                if CUDA:
                    z = z.cuda()
                    y_hot = y_hot.cuda()
                    null_image = null_image.cuda()
                q_null = {'z': z, 'y': y_hot}
                p = dec(null_image, q_null, num_samples=NUM_SAMPLES, batch_size=1)
                image = p['images']
                image = image.value.cpu().detach().numpy().reshape(28, 28)
                axarr[i, j].imshow(image)
                axarr[i, j].axis('off')

        return None

    for style in range(NUM_STYLE):
        vary_z2(style, -3, 3)
        plt.savefig("output/z"+str(style), bbox_inches='tight')
        plt.clf()


def plot_specific_traversals(enc, dec):
    def zi_vs_zj(z_index1, z_index2, zmin=3, zmax=3, num_z=NUM_STYLE, digit=0):
        f, axarr = plt.subplots(num_z, num_z, figsize=(num_z, num_z), sharey=True)
        f.suptitle(r'$Digit: %s$' % digit, fontsize=30)
        z_range = np.linspace(zmin, zmax, num=num_z)

        for i in range(num_z):
            for j in range(num_z):
                null_image = torch.zeros((1, NUM_PIXELS))
                z = torch.zeros((1, NUM_STYLE))
                y_hot = torch.zeros((1, NUM_DIGITS))
                z[0, z_index1] = z_range[i]
                z[0, z_index2] = z_range[j]
                y_hot[0, digit] = 1
                if CUDA:
                    null_image = null_image.cuda()
                    z = z.cuda()
                    y_hot = y_hot.cuda()
                q_null = {'z': z, 'y': y_hot}
                p = dec(null_image, q_null, num_samples=NUM_SAMPLES)
                image = p['images']
                pixels = int(np.sqrt(NUM_PIXELS))
                image = image.value.cpu().detach().numpy().reshape(pixels, pixels)
                axarr[i, j].imshow(image)
                axarr[i, j].axis('off')
        f.text(0.52, 0.08, r'$\mathbf{z_{%d}}$' % z_index2, ha='center', fontsize=20)
        f.text(0.09, 0.5, r'$\mathbf{z_{%d}}$' % z_index1, va='center', rotation='vertical', fontsize=20)

    z_index1 = 9
    z_index2 = 2
    for digit in range(NUM_DIGITS):
        zi_vs_zj(z_index1, z_index2, zmin=-3, zmax=3, num_z=NUM_STYLE, digit=digit)
        plt.figure()


def check_correlations(enc, dec):
    batch_size = 100
    _, data = train.get_data(batch_size)

    Zs = np.zeros((len(data), batch_size, NUM_STYLE))
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == batch_size:
            images = images.view(-1, NUM_PIXELS)
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=1)
            z = q['z'].value.cpu().detach().squeeze().numpy()
            Zs[b] = z
    Zs = Zs.reshape(-1, NUM_STYLE)
    for i in range(NUM_STYLE):
        for j in range(NUM_STYLE):
            if i != j:
                print(i, j, np.corrcoef(Zs[:, i], Zs[:, j])[0, 1])


if __name__ == '__main__':
    load = True
    if load:
        with open("pickled_models.pkl", "rb") as f:
            enc, dec = pickle.load(f)
    else:
        enc, dec = train.train_model(10)
        with open("pickled_models.pkl", "wb") as f:
            pickle.dump((enc, dec), f)

    train_data,test_data = train.get_data(NUM_BATCH)
    print(len(test_data))
    results = train.test(test_data, enc, dec)
    print("results: ")
    print(results)

    plot_style_embeddings(enc, dec)
    plot_traversals(enc, dec)
    # check_correlations(enc, dec)
    plt.show()
