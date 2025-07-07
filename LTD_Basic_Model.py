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
spei_ds = xr.open_dataset('spei_obs_3D.nc')
LTD = LTD_ds['LTD']

# Creating new time coordinate
weekly_time = LTD_ds.time.values

# Converting to weekly (method = linear)
spei_weekly = spei_ds.interp(time=weekly_time, method="linear")

# Function to convert raw values to percentiles
# missing values = -999 in obs - might need to alter for proper percentile (7-4)
def to_percentile(ds, dim='time', missing_val = -999.0):

    valid = ds.where(ds != missing_val)
    # Convert each grid point's time series to percentile values.
    return valid.rank(dim=dim, pct=True)

percentiles_spei = to_percentile(spei_weekly)

# ens = 1, so safe to ignore it from dataset
# testing flattening out entire dataset
spei_df = percentiles_spei.to_dataframe().reset_index()
LTD_df = LTD.to_dataframe().reset_index()
merged_df = pd.merge(spei_df, LTD_df, on=['time', 'lat', 'lon'], how='inner')

# cleaning up NaN entries 
df = merged_df.dropna(subset=['LTD']).copy()

# changing -1 classification to 5 to work with featurespace
df["LTD"] = df["LTD"].replace(-1, 5)

# changing datetime to be usable by TensorFlow, changing LTD from float to int to match example
df["time"] = pd.to_datetime(df["time"]).map(pd.Timestamp.timestamp)


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
        "SPEI1" : "float",
        "SPEI3" : "float",
        "SPEI6" : "float",
        "SPEI12" : "float",
        "SPEI24" : "float",
        "SPEI60" : "float",
        "SPEI2" : "float",
        "SPEI9" : "float",
        "SPEI36" : "float",
        "SPEI48" : "float",
        "SPEI72" : "float",

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
predictions = keras.layers.Dense(6, activation="softmax")(x) # categories: -1, 0, 1, 2, 3, 4

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