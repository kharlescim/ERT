# excerpt of code from ipynb file from Google Colab - mostly for more efficient model training 
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import pandas as pd
import keras
from keras.optimizers import SGD
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import collections

import xarray as xr
import numpy as np

LTD_ds = xr.open_dataset('LTD05.nc')
spei_ds = xr.open_dataset('spei_obs_3D.nc')
obs_ds = xr.open_dataset('obs.nc')

LTD = LTD_ds['LTD']

# Creating new time coordinate
weekly_time = LTD_ds.time.values

# Converting to weekly (method = linear)
spei_weekly = spei_ds.interp(time=weekly_time, method="linear")
obs_weekly = obs_ds.interp(time=weekly_time, method="linear")

# Function to convert raw values to percentiles
# missing values = -999 in obs - might need to alter for proper percentile (7-4)
def to_percentile(ds, dim='time', missing_val = -999.0):

    valid = ds.where(ds != missing_val)
    # Convert each grid point's time series to percentile values.
    return valid.rank(dim=dim, pct=True)

percentiles_spei = to_percentile(spei_weekly)
percentiles_obs = to_percentile(obs_weekly)

# ens = 1, so safe to ignore it from dataset
# testing flattening out entire dataset
spei_df = percentiles_spei.to_dataframe().reset_index()
LTD_df = LTD.to_dataframe().reset_index()
obs_df = (percentiles_obs.to_dataframe().reset_index()).drop(columns=['ens'])
merged_df = pd.merge(spei_df, obs_df, on=['time', 'lat', 'lon'], how='inner')
merged_df = pd.merge(merged_df, LTD_df, on=['time', 'lat', 'lon'], how='inner')

# cleaning up NaN entries 
df = merged_df.dropna().copy()

# changing -1 classification to 5 to work with featurespace
df["LTD"] = df["LTD"].replace(-1, 5)

# changing datetime to be usable by TensorFlow, changing LTD from float to int to match example
df["time"] = pd.to_datetime(df["time"]).map(pd.Timestamp.timestamp)

print(collections.Counter(df["LTD"]))


val_df = df.sample(frac=0.2, random_state=1337)
train_df = df.drop(val_df.index)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("LTD").values.astype("int32")
    features = dataframe.values.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_df).batch(128).prefetch(tf.data.AUTOTUNE)
val_ds = dataframe_to_dataset(val_df).batch(128).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    keras.Input(shape=(35,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(6, activation="softmax"),
])

class_weights = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes=np.unique(df["LTD"]),
    y=df["LTD"]
)
class_weight_dict = dict(enumerate(class_weights))



ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
EarlyStopping = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
 
model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    verbose=2,
    # class_weight=class_weight_dict
    callbacks=[ReduceLROnPlateau, EarlyStopping]
)

y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
y_pred = np.argmax(model.predict(val_ds), axis=1)

print(classification_report(y_true, y_pred, digits=3))
