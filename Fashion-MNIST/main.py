from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from cnn import *
from utils import *

#Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

#Define the output number classes
f_mnist_classes = ['T-shit/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Train/test datasets preparation
train_images, test_images = datasets_resize(train_images, test_images)

#Show a random picked image

model = cnn_arch()
model.summary()


history = model.fit(train_images,
                    train_labels,
                    batch_size=64,
                    epochs=5,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)