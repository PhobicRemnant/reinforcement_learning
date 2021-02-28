from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from cnn import *
from utils import *

#Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

#Define the output number classes
f_mnist_classes = ['T-shit/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define training hyperparameters
epochs=3
batch_size = 250
# Train/test datasets preparation
train_images, test_images = datasets_norm(train_images,test_images)
train_images, test_images = datasets_resize(train_images, test_images)

#Show a random picked image

# Create CNN with no dropout
model = cnn_arch()
model.summary()

# Create CNN with dropout
model_drop = cnn_arch_drop(0.25)
model_drop.summary()

# Train the CNNs
history = train_cnn(model, train_images,train_labels,test_images,test_labels,epochs=epochs,batch_size=batch_size)
plot_acc_history(history, '\\first_model')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# Train the CNNs with dropouts
history = train_cnn(model_drop, train_images,train_labels,test_images,test_labels,epochs=epochs,batch_size=batch_size)
plot_acc_history(history, '\\second_model')

test_loss, test_acc = model_drop.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)