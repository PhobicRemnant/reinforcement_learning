from tensorflow.keras import datasets, layers, models

def cnn_arch():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=16,
                            kernel_size=2,
                            padding='same',
                            activation='relu',
                            input_shape=(28,28,1),
                            name='CONV2D_1'))
    model.add(layers.MaxPool2D(pool_size=2,
                               name='POOL2D_2'))
    model.add(layers.Conv2D(filters=32,
                            kernel_size=2,
                            padding='same',
                            activation='relu',
                            name='CONV2D_3'))
    model.add(layers.MaxPool2D(pool_size=2,
                               name='POOL_4'))
    model.add(layers.Conv2D(filters=64,
                            kernel_size=2,
                            padding='same',
                            name='CONV2D_5',
                            activation='relu'))
    model.add(layers.Flatten(name='FLAT_6'))
    model.add(layers.Dense(128,
                           activation='relu',
                           name='FULL_7'))
    model.add(layers.Dense(10,
                           activation='softmax',
                           name='OUT_8'))
    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def cnn_arch_drop(dropout_percent=0.5):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32,
                            kernel_size=3,
                            padding='same',
                            activation='relu',
                            input_shape=(28,28,1),
                            name='CONV2D_1'))
    model.add(layers.MaxPool2D(pool_size=2,
                               name='POOL2D_2'))
    model.add(layers.Dropout(dropout_percent,
                             name='DROP_3'))
    model.add(layers.Conv2D(filters=64,
                            kernel_size=3,
                            padding='same',
                            activation='relu',
                            name='CONV2D_4'))
    model.add(layers.MaxPool2D(pool_size=2,
                               name='POOL_5'))
    model.add(layers.Dropout(dropout_percent,
                             name='DROP_6'))
    model.add(layers.Conv2D(filters=128,
                            kernel_size=3,
                            padding='same',
                            name='CONV2D_7',
                            activation='relu'))
    model.add(layers.Flatten(name='FLAT_8'))
    model.add(layers.Dense(128,
                           activation='relu',
                           name='FULL_9'))
    model.add(layers.Dense(10,
                           activation='softmax',
                           name='OUT_10'))
    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_cnn(model, train_images, train_labels, test_images, test_labels, epochs, batch_size):
    history = model.fit(train_images,
                        train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(test_images, test_labels))
    return history