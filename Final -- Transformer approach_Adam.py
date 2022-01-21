# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 17:36:30 2021

@author: Shomer
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_pipeline import transformation_pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

# %%
# %% Reading data


data = pd.read_csv('content/preprocessed_train.csv')


# %% Creating a pipeline object and cleaning data


pipeline, data_cleaned = transformation_pipeline(
    data, building_id=122, meter=0, primary_use=99)


# %% Transforming the data and showing it


transformed_data = pipeline.fit_transform(data_cleaned)

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


# %% Creating the model class

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)

# %%


model = build_model(
    (6, 14),  # 6 is for the window on our data 6 hours, and 11 for the features
    head_size=256,  # play with this
    num_heads=8,  # and this
    ff_dim=128,  # and this
    num_transformer_blocks=1,  # and this
    mlp_units=[256],
    mlp_dropout=0.0,
    dropout=0.0,
)

model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
)

model.summary()

# %%
callbacks = [tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)]

model.fit(train_gen, validation_data=val_gen,
          epochs=200,
          callbacks=callbacks,
          )

# %%
model.save('models/Transformer_ADAM')

# %% loading best model
#model = tf.keras.models.load_model('models/transformer_adam')

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

# %%

print('Testing Loss= ', np.mean(tf.keras.losses.MSE(actual, predicted)))
# the mean loss= 0.02899

# %% plotting the validation set output vs the predicted value


fig, (ax1, ax2, ax) = plt.subplots(3, 1,  figsize=(30, 15), sharex=True)

ax1.plot(range(len(actual)),
         predicted,
         color='green', marker='o', linestyle='dashed', label='Predicted')
plt.legend()

ax2.plot(range(len(actual)),
         actual,
         color='red', marker='x', label='Actual')
plt.legend()

ax.plot(range(len(actual)),
        predicted,
        color='green', linestyle='dashed',
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
         color='green', marker='o', linestyle='dashed',
         label='Predicted')

ax2.plot(range(len(actual_t)),
         actual_t,
         color='red', marker='x', label='Actual')


ax.plot(range(len(actual_t)),
        predicted_t,
        color='green', linestyle='dashed',
        label='Predicted')

ax.plot(range(len(actual_t)),
        actual_t,
        color='red',
        label='actual')
plt.title('Train_set', loc='center')

plt.legend()

plt.show()
# In[ ]:
model.summary()
# parameters= 102,595

#%%
