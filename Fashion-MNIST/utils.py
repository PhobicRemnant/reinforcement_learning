import matplotlib.pyplot as plt

def plot_samples(images):

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(330 + 1 + i)
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))

    plt.show()
    return

def dataset_norm(dataset):
    return dataset.astype('float32')/255

def dataset_resize(dataset):
    dataset = dataset.resize((dataset.shape[0], 28, 28, 1))
    return dataset