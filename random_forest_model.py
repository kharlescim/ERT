# excerpt of code from ipynb file from Google Colab - mostly for more efficient model training 

import pandas as pd
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

import xarray as xr
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

LTD_ds = xr.open_dataset('LTD05.nc')
spei_ds = xr.open_dataset('spei_obs_3D.nc')
obs_ds = xr.open_dataset('obs.nc')
usdm_ds = xr.open_dataset('USDM05_2000_2024.nc')

LTD = LTD_ds['LTD']
usdm = usdm_ds['USDM']

# Creating new time coordinate
weekly_time = usdm.time.values

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
usdm_df = usdm.to_dataframe().reset_index()
obs_df = (percentiles_obs.to_dataframe().reset_index()).drop(columns=['ens'])
LTD_df = LTD.to_dataframe().reset_index()
merged_df = pd.merge(spei_df, obs_df, on=['time', 'lat', 'lon'], how='inner')
merged_df = pd.merge(merged_df, LTD_df, on=['time', 'lat', 'lon'], how='inner')
merged_df = pd.merge(merged_df, usdm_df, on=['time', 'lat', 'lon'], how='inner')


# selecting important features, dropping NaN values
df = merged_df[["time", "lat", "lon", "USDM", "SPEI9", "SPEI12", "SMP3", "SPEI6", "SPI12", "SPI9", "SMP6", "SRI6", "SMP1", "SPI6", "SRI9", "SRI3", "SPEI3", "SMP9", "SPEI24", "SRI12", "SPI24", "LTD"]]
df = df.dropna()


# changing datetime to be usable by TensorFlow, changing LTD from float to int to match example
df["time"] = pd.to_datetime(df["time"]).map(pd.Timestamp.timestamp)


val_df = df.sample(frac=0.2, random_state=1337)
train_df = df.drop(val_df.index)


# Split features and labels
X_train = train_df.drop(columns=["USDM"]).values
y_train = train_df["USDM"].values

X_val = val_df.drop(columns=["USDM"]).values
y_val = val_df["USDM"].values
model = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, digits=3))

joblib.dump(model, "C:/Users/kitti/ERT/rf_usdm_model.joblib")
