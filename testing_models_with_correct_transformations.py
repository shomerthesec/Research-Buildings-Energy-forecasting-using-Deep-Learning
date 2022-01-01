# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:06:03 2022

@author: Shomer
"""


# In[Importing libraries]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_pipeline import transformation_pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf


# %% Reading data


data = pd.read_csv('content/preprocessed_train.csv')

# let's pick the data wiht primary_use==0
data = data.query('primary_use==0 & meter==0')

# %%
b_id = data.building_id.unique()

# %%
# a function for plotting


def plot_output(actual, predicted, title, building_idx, avg_loss):
    fig, (ax1, ax2, ax) = plt.subplots(3, 1,  figsize=(30, 15), sharex=True)

    fig.suptitle(
        f'{title} model for building {building_idx}\nmse={avg_loss:.5f}', fontsize=24)

    ax1.plot(range(len(actual)),
             predicted,
             color='green', linestyle='dashed')
    ax1.set_title('Predicted')
    ax1.set_ylim(0, 1)

    ax2.plot(range(len(actual)),
             actual,
             color='red', label='Actual')
    ax2.set_title('Actual')
    ax2.set_ylim(0, 1)

    ax.plot(range(len(actual)),
            predicted,
            color='green', linestyle='dashed',
            label='Predicted')

    ax.plot(range(len(actual)),
            actual,
            color='red',
            label='actual')
    ax.set_ylim(0, 1)
    plt.legend()
    plt.show()
    fig.savefig(
        f'Plots/{title}_model_correct/building {building_idx} -- mse {avg_loss:.5f}.png')

# %%

# function to laod certain building id


def loading_data(idx):
    pipeline, data_cleaned = transformation_pipeline(
        data, building_id=idx, meter=0, primary_use=0)

    train, test = train_test_split(data_cleaned,
                                   # [:, 1:],
                                   #transformed_data[:, 0],
                                   train_size=0.2,
                                   shuffle=False,
                                   random_state=2021)

    train_data = pipeline.fit_transform(train)
    test_data = pipeline.transform(test)

    x_train, y_train = train_data[:, 1:], train_data[:, 0]

    x_val, y_val = test_data[:, 1:], test_data[:, 0]

    train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_train,
                                                                    y_train,
                                                                    length=6, sampling_rate=1,
                                                                    stride=1, batch_size=32,
                                                                    shuffle=False
                                                                    )

    val_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_val,
                                                                  y_val,
                                                                  length=6, sampling_rate=1,
                                                                  stride=1, batch_size=32,
                                                                  shuffle=False
                                                                  )
    return train_gen, val_gen


# %% loading models to evaluate them on new data
models = ['models/Transformer_adam',
          'models/GRU_ADAM',
          'models/LSTM_ADAM']
for building_idx in b_id:
    train_gen, test_gen = loading_data(building_idx)

    for model_address in models:
        predicted = []
        actual = []

        model = tf.keras.models.load_model(model_address)
        model.fit(train_gen, epochs=15, verbose=False)

        for i in range(50):
            predicted.extend(model.predict(test_gen[i][0]))
            actual.extend(test_gen[i][1])

        txt = model_address.split('/')[1].split('_')[0]
        avg_loss = np.mean(tf.keras.losses.MSE(actual, predicted))

        plot_output(actual, predicted, txt, building_idx, avg_loss)

# %%
