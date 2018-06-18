from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import os
import tensorflow as tf
import keras.backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

curr_dir = os.chdir("C:/Users/PC/Desktop/Fruit Classification/Data/Training")
curr_dir = os.getcwd()

classes = [name for name in os.listdir(".") if os.path.isdir(name)]
classifier = load_model(
    "C:/Users/PC/Desktop/Fruit Classification/weights/weights-improvement-50-0.9861.hdf5")
datagen_test = ImageDataGenerator(rescale=1./255)
test_dir = "C:/Users/PC/Desktop/Fruit Classification/Data/Test/"
generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  batch_size=12,
                                                  target_size=(
                                                      100, 100),
                                                  color_mode='rgb',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  )
y_test = generator_test.classes
y_pred = []
result = classifier.predict_generator(generator_test)
res_dic_list = []
for _ in range(len(classes)):
    res_dic_list.append(dict())
for i, img in enumerate(result):
    name = classes[np.argmax(img)]
    y_pred.append(np.argmax(img))
    true_class_index = y_test[i]
    true_name = classes[true_class_index]

    if name in res_dic_list[true_class_index]:
        res_dic_list[true_class_index][name] = res_dic_list[true_class_index][name] + 1
    else:
        res_dic_list[true_class_index][name] = 1

for i in range(len(classes)):
    print(classes[i])
    print(res_dic_list[i])
    print('\n')

from sklearn.metrics import classification_report, confusion_matrix

target_names = classes
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))
