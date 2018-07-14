import os
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import cv2 
from scipy.stats import norm

from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Dropout, Conv2D, Flatten, UpSampling2D, MaxPooling2D, Reshape
from keras.models import Model, Sequential

import glob
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

from PIL import Image, ImageFile, ImageFilter


# Reading dataset 

x_train = []

def read_dataset(folder_name):
    dir_or_files = os.listdir(folder_name)
    for dir_or_file in dir_or_files:
        print(dir_or_file)
        if os.path.isdir(os.path.join(folder_name,dir_or_file)):
            read_dataset(os.path.join(folder_name, dir_or_file))
            # print("if check")
        else:
            img = cv2.imread(os.path.join(folder_name, dir_or_file))
            print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img =img/255
            print(img.shape)
            height, width = img.shape[:2]
            img = cv2.resize(img,(128, 192), interpolation = cv2.INTER_CUBIC) 
            img = img[:, :, newaxis]
            print(img.shape)
            x_train.append(img)
            # total_data_labels.append(os.path.join(folder_name, dir_or_file))

read_dataset('/home/dlagroup5/VAE/Fingerprint/Phase2/Lum')
# print(total_data.shape)
x_train_size = len(x_train)
x_train = np.array(x_train)
print(x_train.shape)

# Network Parameters 

original_dim = (x_train.shape[1], x_train.shape[2], 1)
print(original_dim)
latent_dim = (64,)
epsilon_std = 1
epochs = 20
batch_size = 32

# Reshaping the input image into required shape 
# x_train = np.reshape(x_train, (x_train.shape[0], original_dim[0], original_dim[1], original_dim[2]))


def nll(y_true, y_pred):
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

# Encoder 

x = Input(shape=original_dim)

h = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D()(h)

h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D()(h)

h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D()(h)

h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D()(h)

h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D()(h)

h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
h = MaxPooling2D()(h)

h = Flatten()(h)
h = Dense(1024)(h)
h = Dropout(0.5)(h) 

# Latent vector 
z_mu = Dense(latent_dim[0])(h)
z_log_var = Dense(latent_dim[0])(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(0.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim[0])))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])


# Decoder 

# decoder = Sequential([
    #Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    #Dense(original_dim, activation='sigmoid')
    # Dense(1024, input_shape=latent_dim),
    # Dense((14 * 9 * 256)),
    # Dropout(0.5),
    # Reshape((14, 9, 256)),

    # UpSampling2D(),
    # Conv2D(256, (3, 3), activation='relu', padding='same'),
    # Conv2D(256, (3, 3), activation='relu', padding='same'),
    # Conv2D(256, (3, 3), activation='relu', padding='same'),

    # UpSampling2D(),
    # Conv2D(128, (3, 3), activation='relu', padding='same'),
    # Conv2D(128, (3, 3), activation='relu', padding='same'),
    # Conv2D(128, (3, 3), activation='relu', padding='same'),

    # UpSampling2D(),
    # Conv2D(64, (3, 3), activation='relu', padding='same'),
    # Conv2D(64, (3, 3), activation='relu', padding='same'),
    # Conv2D(64, (3, 3), activation='relu', padding='same'),

    # UpSampling2D(),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(1, (3, 3), activation='sigmoid', padding='same'),

    # UpSampling2D(),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(1, (3, 3), activation='sigmoid', padding='same'),

    # UpSampling2D(),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Conv2D(1, (3, 3), activation='sigmoid', padding='same')
# ])
d= Dense(1024, input_shape=latent_dim)(z)
d= Dense((3 * 2 * 256))(d)
d= Dropout(0.5)(d)
d= Reshape((3, 2, 256))(d)

d= UpSampling2D()(d)
d= Conv2D(256, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(256, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(256, (3, 3), activation='relu', padding='same')(d)

d= UpSampling2D()(d)
d= Conv2D(128, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(128, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(128, (3, 3), activation='relu', padding='same')(d)

d= UpSampling2D()(d)
d= Conv2D(64, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(64, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(64, (3, 3), activation='relu', padding='same')(d)

d= UpSampling2D()(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

d= UpSampling2D()(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

d= UpSampling2D()(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
d= Conv2D(32, (3, 3), activation='relu', padding='same')(d)
x_pred= Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)
# x_pred = Lambda(reshape,)
# x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)
vae.summary()

# x_train, x_test, _ , _ = train_test_split(x_train, x_train, test_size=0.2, random_state=42)

# Train and save 
from keras.callbacks import EarlyStopping
for epoch in range(0, 5):
    #model.fit_generator(train_gen, epochs=20, steps_per_epoch=len(img_paths) / 100, callbacks=[EarlyStopping(monitor='loss', patience=3)], verbose=True, use_multiprocessing=True)
    #model.evaluate_generator(validation_gen, use_multiprocessing=True)
    vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2)
    vae.save("vae_" + str(epoch) + ".h5")

encoder = Model(x, z_mu)
encoder.save("enc.h5")
decoder.save("dec.h5")


