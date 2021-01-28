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

def datasets_resize(train_dataset, test_dataset):
    train_dataset = train_dataset.reshape((train_dataset.shape[0], 28, 28, 1))
    test_dataset = test_dataset.reshape((test_dataset.shape[0], 28, 28, 1))
    return train_dataset, test_dataset