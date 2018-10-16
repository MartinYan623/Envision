import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Activation , Dropout
from sklearn.metrics import mean_squared_error
from keras.optimizers import RMSprop, Adam
from numpy.random import seed
from common_misc import load_data_from_pkl


# Create model
def create_fc_model():
    model = Sequential([
        Dense(30, input_dim=15, kernel_initializer='RandomNormal'),
        Activation('elu'),
        Dropout(rate=0.4, seed=True),
        Dense(10),
        Activation('elu'),
        Dense(1)
    ])
    return model

# Create model
def create_fc_model2():
    model = Sequential()
    model.add(
        SimpleRNN(20, stateful=False, return_sequences=False, batch_input_shape=(1, 15, 1), activation='relu'))
    model.add(Dense(1))
    return model


x_train, y_train = load_data_from_pkl('data/turbine_1_train.pkl')
x_test, y_test = load_data_from_pkl('data/turbine_1_test.pkl')

data_train = pd.concat([x_train, y_train], axis=1)
data_test = pd.concat([x_test, y_test], axis=1)

# drop out nan value
data_train = data_train.dropna(subset=['Y.ws_tb'])
data_train = data_train[np.isnan(data_train['GFS0.ws']) == False]
data_train = data_train[np.isnan(data_train['WRF0.ws']) == False]
data_test = data_test.dropna(subset=['Y.ws_tb'])

x_train = data_train[['EC0.ws','EC0.wd','EC0.tmp','EC0.rho','EC0.pres',
'GFS0.ws','GFS0.wd','GFS0.tmp','GFS0.rho','GFS0.pres',
'WRF0.ws','WRF0.wd','WRF0.tmp','WRF0.rho','WRF0.pres']]
count1=len(x_train)
print(count1)
y_train=data_train['Y.ws_tb']
x_test=data_test[['EC0.ws','EC0.wd','EC0.tmp','EC0.rho','EC0.pres',
'GFS0.ws','GFS0.wd','GFS0.tmp','GFS0.rho','GFS0.pres',
'WRF0.ws','WRF0.wd','WRF0.tmp','WRF0.rho','WRF0.pres']]
count2=len(x_test)
y_test=data_test['Y.ws_tb']

"""
x_train=x_train.values
x_train = x_train.reshape(count1,15,1)
y_train=y_train.values
y_train = y_train.reshape(count1,1)
x_test=x_test.values
x_test = x_test.reshape(count2,15,1)
y_test=y_test.values
y_test = y_test.reshape(count2,1)
"""

print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)


epoches=20
# Create the model
print('Creating Fully-Connected Model...')
model_fc = create_fc_model()
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.00005)
model_fc.compile(optimizer=adam, loss='mean_squared_error')
# Train the model
print('Training')
##### TRAIN YOUR MODEL #####
history = model_fc.fit(x_train, y_train, epochs=epoches, batch_size=1, validation_data=(x_test, y_test), shuffle=False)

# Plot and save loss curves of training and test set vs iteration in the same graph
##### PLOT AND SAVE LOSS CURVES #####
loss = history.history['loss']
val_loss = history.history['val_loss']

predicted_fc = model_fc.predict(x_test, batch_size=1)
##### CALCULATE RMSE #####
fc_rmse = np.sqrt(mean_squared_error(y_test, predicted_fc))
print(fc_rmse)

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, epoches+1), loss, label='train_loss')
plt.plot(np.arange(1, epoches+1), val_loss, label='val_loss')
plt.title('Loss vs Iterations in Training and Validation Set')
plt.xlabel('Iterations')
plt.ylabel('Loss')
x_label = range(1, epoches+1)
plt.xticks(x_label)
plt.legend()
plt.grid()
plt.show()

