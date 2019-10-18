from __future__ import print_function, division
import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.optimizers import SGD, adam
from pandas import read_csv


def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):

    model = Sequential((

        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(window_size, nb_input_series)),
        MaxPooling1D(),     # Downsample the output of convolution by 2X.
        Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
        MaxPooling1D(),
	    Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu'),
		MaxPooling1D(),
		Flatten(),
        Dense(nb_outputs, activation='linear'),  # For binary classification, change the activation to 'sigmoid'
    ))
    sgd = SGD(lr=10e-6)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    return model


def make_timeseries_instances(timeseries, window_size):
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    y = timeseries[window_size:]
    q = np.atleast_3d([timeseries[-window_size:]])
    return X, y, q


def evaluate_timeseries(timeseries, window_size):

    filter_length = 5
    nb_filter = 4
    timeseries = np.atleast_2d(timeseries)
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T       # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=nb_series, nb_outputs=nb_series, nb_filter=nb_filter)
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
    model.summary()

    X, y, q = make_timeseries_instances(timeseries, window_size)
    print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q, sep='\n')
    test_size = int(0.05 * nb_samples)           # In real life you'd want to use 0.2 - 0.5
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    model.fit(X_train, y_train, nb_epoch=500, batch_size=50, validation_data=(X_test, y_test))

    pred = model.predict(X_test)
    print('\n\nactual', 'predicted', sep='\t')
    for actual, predicted in zip(y_test, pred.squeeze()).__iter__():
        print(actual.squeeze(), predicted, sep='\t')
    print('next', model.predict(q).squeeze(), sep='\t')

timeseries = read_csv('well_16_i.csv', usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], engine='python', skipfooter=3)
timeseries = timeseries.drop(timeseries.index[0])
timeseries[timeseries < 0] = np.nan
timeseries = timeseries.replace(to_replace=np.nan, value=0, method='pad')
timeseries = timeseries.values
timeseries = timeseries.astype('float32')
def main(timeseries):
    np.set_printoptions(threshold=15)
    ts_length = timeseries
    window_size = 40
    print('\nSimple single timeseries vector prediction')
    timeseries = np.array(ts_length)                   # The timeseries f(t) = t
    evaluate_timeseries(timeseries, window_size)

    print('\nMultiple-input, multiple-output prediction')
    timeseries = np.array([np.array(ts_length), -np.array(ts_length)]).T      # The timeseries f(t) = [t, -t]
    evaluate_timeseries(timeseries, window_size)


if __name__ == '__main__':
    main(timeseries)