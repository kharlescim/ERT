# excerpt of code from ipynb file from Google Colab - mostly for more efficient model training 
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import pandas as pd
import keras
from keras.utils import FeatureSpace

import xarray as xr
import numpy as np

LTD_ds = xr.open_dataset('LTD05.nc')
USDM05_ds = xr.open_dataset('USDM05_2000_2024.nc')

# LTD and USDM only have one variable, so they can be treated as DataArrays
LTD = LTD_ds['LTD']
USDM = USDM05_ds['USDM']

LTD_df = LTD.to_dataframe().reset_index()
# print(np.sort(LTD_df['LTD'].unique()))

# cleaning up NaN entries and -1 entries 
df = LTD_df.dropna(subset=['LTD']) 
df = df[df['LTD'] != -1.0]

#visualizing 
# print(df['LTD'].value_counts().sort_index())


# changing datetime to be usable by TensorFlow, changing LTD from float to int to match example 
df["time"] = pd.to_datetime(df["time"]).map(pd.Timestamp.timestamp)
df["LTD"] = df["LTD"].astype(int)
df.head() # LTD is target 

val_df = df.sample(frac=0.2, random_state=1337)
train_df = df.drop(val_df.index)

'''print(
    "Using %d samples for training and %d for validation"
    % (len(train_df), len(val_df))
)'''

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("LTD")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_df)
val_ds = dataframe_to_dataset(val_df)

'''for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)'''

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

feature_space = FeatureSpace(
    features={
        "lat": "float",
        "lon": "float",
        "time": "float",
    },
    output_mode="concat"
)

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    # print("preprocessed_x.shape:", preprocessed_x.shape)
    # print("preprocessed_x.dtype:", preprocessed_x.dtype)

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(32, activation="relu")(encoded_features)
x = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(5, activation="softmax")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)
training_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)

# val_accuracy is plateauing wayyy too early at .2678 
# looks like model can't learn useful patterns from time, lat and lon alone - which is kind of expected 
training_model.fit(
    preprocessed_train_ds,
    epochs=20,
    validation_data=preprocessed_val_ds,
    verbose=2,
)