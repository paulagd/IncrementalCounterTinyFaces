import cv2
from keras.layers import Dense, Input
from keras.regularizers import l2
import pickle
import numpy as np
import math
from keras.models import Model
import keras
import tensorflow as tf
import timeit
from keras.utils.vis_utils import plot_model


class CrowdCounting():

    def __init__(self):
        self.model = None

        self.init()

    def init(self):
        # load neural network architecture
        Res50 = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3),
                                                     pooling="avg")
        input_a = Input(shape=(224, 224, 3))
        Res50.trainable = False
        shared_a = Res50(input_a)
        reg = l2(l=5e-4)
        dense1 = Dense(100, activation="relu", W_regularizer=reg)(shared_a)
        dense2 = Dense(100, activation="relu", W_regularizer=reg)(dense1)
        dense3 = Dense(50, activation="relu", W_regularizer=reg)(dense2)
        dense4 = Dense(50, activation="relu", W_regularizer=reg)(dense3)
        dense5 = Dense(1, activation="relu", W_regularizer=reg)(dense4)

        # add corresponding weights for crowd_counting model
        net_crowd = Model(input=input_a, output=[dense5])
        # plot_model(net_crowd, to_file='model_plot_CROWDS.png', show_shapes=True, show_layer_names=True)

        net_crowd.load_weights("./weights/counting_weights.hdf5")

        self.filename = 0

        self.model = net_crowd
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def process_img(self, im):
        ''' pre-process image to feed the crowd counting model '''

        start_time = timeit.default_timer()

        a = im.shape[0] % 100
        b = im.shape[1] % 100

        scale = np.pad(im, [(100 - a, 0), (0, 0)], mode="constant", constant_values=[0]);
        scale = np.pad(scale, [(0, 0), (100 - b, 0)], mode="constant", constant_values=[0]);

        w = math.floor(scale.shape[0] / 100.)
        h = math.floor(scale.shape[1] / 100.)

        count_sum = 0

        for i in range(1, int(w)):
            for j in range(1, int(h)):
                x = i * 100
                y = j * 100

                crop = scale[x - 100:x, y - 100:y]
                crop = cv2.resize(crop, (224, 224))
                crop = np.stack((crop,) * 3)
                crop = np.expand_dims(crop, axis=0)
                crop = crop.transpose(0, 2, 3, 1)

                # predict the crowd_counting
                count_sum += self.model.predict(crop)

        end_time = timeit.default_timer()

        return int(count_sum[0][0]), (end_time - start_time)

    def run(self, data):

        img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        count = 0

        # to make tensor flow model persistent to multithreading
        with self.graph.as_default():
            global count
            count = self.process_img(img)


        return count
