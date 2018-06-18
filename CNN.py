import tensorflow as tf
import keras.backend as K
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

train_dir = "C:/Users/PC/Desktop/Fruit Classification/Data/Training"
test_dir = "C:/Users/PC/Desktop/Fruit Classification/Data/Test"
model_weight_path = os.path.join(
    "C:/Users/PC/Desktop/Fruit Classification/weights", "weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5")

curr_dir = os.chdir("C:/Users/PC/Desktop/Fruit Classification/Data/Training")
curr_dir = os.getcwd()

classes = [name for name in os.listdir(".") if os.path.isdir(name)]
print(classes)

img_size = 100
channel = 3
epochs = 50
batch_size = 80
class_mode = 'categorical'
color_mode = 'rgb'

datagen_train = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    batch_size=batch_size,
                                                    target_size=(
                                                        img_size, img_size),
                                                    shuffle=True,
                                                    class_mode=class_mode,
                                                    color_mode=color_mode)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  batch_size=batch_size,
                                                  target_size=(
                                                      img_size, img_size),
                                                  color_mode=color_mode,
                                                  class_mode=class_mode,
                                                  shuffle=False)

y_train = generator_train.classes

class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
print(class_weight)

pre_computed_weights = dict(zip(classes, class_weight))
print("Pre computed weights for each class : \n", pre_computed_weights)

steps_test = generator_test.n // batch_size
print("The steps for batch size {} of test set is {}".format(batch_size, steps_test))

steps_per_epoch = generator_train.n // batch_size
print("The steps for batch size {} of training set is {}".format(
    steps_per_epoch, steps_per_epoch))


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(img_size, img_size, channel)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(5000, activation='relu'))
model.add(Dense(len(classes), activation='sigmoid'))

model.summary()

opt = keras.optimizers.SGD(lr=0.001, nesterov=True)
model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
check_point = keras.callbacks.ModelCheckpoint(
    model_weight_path, monitor='val_acc', save_best_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10)

callback_list = [check_point, reduce_lr]


model = model.fit_generator(generator_train,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=generator_test,
                            callbacks=callback_list,
                            validation_steps=steps_test,
                            class_weight=class_weight)


def plot_model_history(model):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    ax[0].plot(range(1, len(model.history['acc']) + 1), model.history['acc'])
    ax[0].plot(range(1, len(model.history['val_acc']) + 1),
               model.history['val_acc'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_xticks(np.arange(1, len(model.history['acc']) + 1),
                     len(model.history['acc']) / 10)
    ax[0].legend(['train', 'val'], loc='best')

    ax[1].plot(range(1, len(model.history['loss']) + 1), model.history['loss'])
    ax[1].plot(range(1, len(model.history['val_loss']) + 1),
               model.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_xticks(np.arange(1, len(model.history['loss']) + 1),
                     len(model.history['loss']) / 10)
    ax[1].legend(['train', 'val'], loc='best')

    plt.show()


plot_model_history(model)
