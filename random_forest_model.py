import pandas as pd
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import collections
import xarray as xr
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

df = df.drop(columns = ["SPI9", "SRI6", "SPEI48", "SRI24", "SMP1", "SMP24", "SPEI6", "SPEI60", "SPEI72", "SRI3", "SPI60", "SPI6", "SMP60", "SRI1", "SPEI3", "SRI60", "SPEI2", "SPI3", "SPEI1", "SMP3", "SPEI36", "SPI24"])

val_df = df.sample(frac=0.2, random_state=1337)
train_df = df.drop(val_df.index)

# Split features and labels
X_train = train_df.drop(columns=["LTD"]).values
y_train = train_df["LTD"].values

X_val = val_df.drop(columns=["LTD"]).values
y_val = val_df["LTD"].values
model = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, digits=3))