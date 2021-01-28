from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from cnn import *
from utils import *

#Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

#Define the output number classes
f_mnist_classes = ['T-shit/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize train and test datasets
#test_images = test_images.astype('float32') / 255
#train_labels = train_labels.astype('float32') / 255

train_images = train_images.reshape(train_images.shape[0],28,28,1)
#test_images = test_images.reshape(test_images.shape[0],28,28,1)

#train_images = dataset_resize(train_images)
testX = dataset_resize(test_images)

print(train_images.shape)
print(testX)


#Show a random picked image
#show_img = random.choice(train_images)

#model = cnn_arch()
#model.summary()
"""
history = model.fit(train_images,
                    train_labels,
                    batch_size=64,
                    epochs=3,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
"""