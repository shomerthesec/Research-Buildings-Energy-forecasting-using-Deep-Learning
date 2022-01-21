# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:25:41 2022

@author: Shomer
"""

# In[Importing libraries]:

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_pipeline import transformation_pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# %% Reading data


data = pd.read_csv('content/preprocessed_train.csv')

# let's pick the data wiht primary_use==0
data = data.query('primary_use==0 & meter==0')

# %%
b_id = [118, 122, 125]

# %%
# a function for plotting


def plot_output(actual, predicted, title, building_idx, avg_loss, avg_rmse):
    fig, (ax1, ax2, ax) = plt.subplots(3, 1,  figsize=(30, 15), sharex=True)

    fig.suptitle(
        f'{title} model for building {building_idx}\nmse={avg_loss:.5f}\nrmse={avg_rmse:.5f}', fontsize=24)

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
        f'Plots/Fine-tuning models 20% data/{title} model -- building {building_idx} -- mse {avg_loss:.5f} -- rmse {avg_rmse:.5f}.png')

# %%

# function to laod certain building id


def loading_data(idx):
    pipeline, data_cleaned = transformation_pipeline(
        data, building_id=idx, meter=0, primary_use=0)

    train, test = train_test_split(data_cleaned,
                                   # [:, 1:],
                                   #transformed_data[:, 0],
                                   test_size=0.2,
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
                                                                  stride=1, batch_size=350,
                                                                  shuffle=False
                                                                  )
    return train_gen, val_gen[0]


# %% loading pretrained models to finetune them for 15 epochs
models = ['models/Transformer_adam',
          'models/GRU_ADAM',
          'models/LSTM_ADAM']
finetuning_data = {}
for building_idx in b_id:
    train_gen, test_gen = loading_data(building_idx)

    finetuning_data[building_idx] = {}

    for model_address in models:
        predicted = np.array([])
        actual = np.array([])

        txt = model_address.split('/')[1].split('_')[0]

        finetuning_data[building_idx][txt] = []

        model = tf.keras.models.load_model(model_address)

        start_time = time.time()

        model.fit(train_gen, epochs=15, verbose=False)
        time_taken = time.time() - start_time
        print(f"model {txt} took {time.time() - start_time} seconds")

        predicted = np.append(predicted, model.predict(test_gen[0]))
        actual = np.append(actual, test_gen[1])

        avg_mse = np.mean((actual - predicted)**2)
        avg_rmse = np.sqrt(np.mean((actual - predicted)**2))

        finetuning_data[building_idx][txt].append(
            (avg_mse, avg_rmse, time_taken))

        plot_output(actual, predicted, txt, building_idx, avg_mse, avg_rmse)
# %%
transformer, gru, lstm = [], [], []
for building_idx in b_id:

    x, y, z = finetuning_data[building_idx].values()
    transformer.append(x)
    gru.append(y)
    lstm.append(z)

# %%
mse, rmse, t = [], [], []
for d in [transformer, gru, lstm]:
    for i in range(3):

        mse.append(d[i][0][0])
        rmse.append(d[i][0][1])
        t.append(d[i][0][2])

    d.append([np.mean(mse), np.mean(rmse), np.mean(t)])

# %%
for d, n in zip([transformer, gru, lstm], ['transformer', 'gru', 'lstm']):
    print(f'--- {n} ---')
    print(' mse \t rmse \t time')
    print(f'{d[3][0]:0.4f}\t{d[3][1]:0.4f}\t{d[3][2]:0.4f}')
