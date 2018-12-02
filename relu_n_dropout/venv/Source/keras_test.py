import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def keras_test():
    logr = Sequential()
    logr.add(Dense(1, input_dim=2, activation='sigmoid'))
    logr.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    def sampler(n, x, y):
        return np.random.normal(size=[n, 2]) + [x, y]

    def sample_data(n=1000, p0=(-1., -1.), p1=(1., 1.)):
        zeros, ones = np.zeros((n, 1)), np.ones((n, 1))
        labels = np.vstack([zeros, ones])
        z_sample = sampler(n, x=p0[0], y=p0[1])
        o_sample = sampler(n, x=p1[0], y=p1[1])
        return np.vstack([z_sample, o_sample]), labels

    x_train, y_train = sample_data()
    x_test, y_test = sample_data(100)

    logr.fit(x_train, y_train, batch_size=16, epochs=100, verbose=1, validation_data=(x_test, y_test))

keras_test()
