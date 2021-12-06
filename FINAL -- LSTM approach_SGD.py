
# -*- coding: utf-8 -*-
# In[Importing libraries]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_pipeline import transformation_pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf


# %% Reading data


data = pd.read_csv('content/preprocessed_train.csv')


# %% Creating a pipeline object and cleaning data


pipeline, data_cleaned = transformation_pipeline(
    data, building_id=122, meter=0, primary_use=99)


# %% Transforming the data and showing it


transformed_data = pipeline.fit_transform(data_cleaned)
display(pd.DataFrame(transformed_data, index=data_cleaned.index,
        columns=data_cleaned.columns).head())


# %% displaying the meter reading


display(transformed_data[:, 0])  # this gives us the meter reading


# %% Showing the rest of the data

display(transformed_data[:, 1:])  # this gives us the rest of the columns


# %% Splitting the data


x_train, x_val, y_train, y_val = train_test_split(transformed_data[:, 1:],
                                                  transformed_data[:, 0],
                                                  test_size=0.2,
                                                  shuffle=False,
                                                  random_state=2021)


# %% Creating time series data generators


train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_train,
                                                                y_train,
                                                                length=6, sampling_rate=1,
                                                                stride=1, batch_size=32
                                                                )

val_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_val,
                                                              y_val,
                                                              length=6, sampling_rate=1,
                                                              stride=1, batch_size=32
                                                              )


# %% Creating the model


model = tf.keras.Sequential([tf.keras.layers.LSTM(128, activation='relu', return_sequences=False),
                            tf.keras.layers.Dense(1)])


# %% Training the model


model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(0.0001))

cb = tf.keras.callbacks.ModelCheckpoint(filepath='models/LSTM_SGD',
                                        monitor='val_loss',
                                        verbose=0, save_best_only=True)
# Fitting the model
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=200,
                    callbacks=[cb],
                    shuffle=False)

# %% loading best model
model = tf.keras.models.load_model('models/LSTM_SGD')

# %% Displaying 1 batch of the validation data


# where 7 is the batch , 0 stands for the features and 1 stands for the output
display(val_gen[7][1])


# %% predicting that batch


predicted_batch_7 = model.predict(val_gen[7][0])


# %% plotting the prediction vs the actial


_, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(32),
        predicted_batch_7,
        color='green', label='Predicted')

ax.plot(range(32),
        val_gen[7][1],
        color='red', label='Actual')
ax.legend()

plt.show()
# %% Predicting the whole batches


# lets try predicting more than one patch

predicted = []
actual = []
for i in range(32):
    predicted.extend(model.predict(val_gen[i][0]))
    actual.extend(val_gen[i][1])


# %% plotting the validation set output vs the predicted value


fig, (ax1, ax2, ax) = plt.subplots(3, 1,  figsize=(30, 15), sharex=True)

ax1.plot(range(len(actual)),
         predicted,
         color='green')
plt.legend()

ax2.plot(range(len(actual)),
         actual,
         color='red')
plt.legend()

ax.plot(range(len(actual)),
        predicted,
        color='green',
        label='Predicted')
plt.legend()
ax.plot(range(len(actual)),
        actual,
        color='red',
        label='actual')

plt.legend()

plt.title('Test_set', loc='center')

plt.show()
# %% Let's try to see the effect on the training data

predicted_t = []
actual_t = []
for i in range(32):
    predicted_t.extend(model.predict(train_gen[i][0]))
    actual_t.extend(train_gen[i][1])

# %% plotting the result
fig, (ax1, ax2, ax) = plt.subplots(3, 1,  figsize=(30, 15), sharex=True)

ax1.plot(range(len(actual_t)),
         predicted_t,
         color='green')

ax2.plot(range(len(actual_t)),
         actual_t,
         color='red')


ax.plot(range(len(actual_t)),
        predicted_t,
        color='green',
        label='Predicted')

ax.plot(range(len(actual_t)),
        actual_t,
        color='red',
        label='actual')
plt.title('Traine_set', loc='center')

plt.legend()

plt.show()
# In[ ]:
