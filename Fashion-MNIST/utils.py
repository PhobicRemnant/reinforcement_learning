import matplotlib.pyplot as plt
import os

main_folder = os.getcwd()
graphics_folder = os.getcwd() + '\\graphics'

def plot_samples(images):

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(330 + 1 + i)
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))

    plt.show()


def datasets_norm(x_dataset, y_dataset):

    x_dataset = x_dataset.astype('float32')/255
    y_dataset = y_dataset.astype('float32')/255
    return x_dataset, y_dataset

def datasets_resize(train_dataset, test_dataset):

    train_dataset = train_dataset.reshape((train_dataset.shape[0], 28, 28, 1))
    test_dataset = test_dataset.reshape((test_dataset.shape[0], 28, 28, 1))
    return train_dataset, test_dataset

def plot_acc_history(history, model_name):

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig(graphics_folder + model_name)
    plt.show()
