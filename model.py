import pandas as pd
import xarray as xr
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


LTD_ds = xr.open_dataset('LTD05.nc')
spei_ds = xr.open_dataset('spei_obs_3D.nc')
obs_ds = xr.open_dataset('obs.nc')
usdm_ds = xr.open_dataset('USDM05_2000_2024.nc')

LTD = LTD_ds['LTD']
usdm = usdm_ds['USDM']

# creating new time coordinate
weekly_time = usdm.time.values

# converting to weekly (method = linear)
spei_weekly = spei_ds.interp(time=weekly_time, method="linear")
obs_weekly = obs_ds.interp(time=weekly_time, method="linear")

# function to convert raw values to percentiles
# missing values = -999 in obs 
def to_percentile(ds, dim='time', missing_val = -999.0):

    valid = ds.where(ds != missing_val)
    # Convert each grid point's time series to percentile values.
    return valid.rank(dim=dim, pct=True)

percentiles_spei = to_percentile(spei_weekly)
percentiles_obs = to_percentile(obs_weekly)

# ens = 1, so safe to ignore it from dataset
# flattening out datasets
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


# changing datetime to be usable by model
df["time"] = pd.to_datetime(df["time"]).map(pd.Timestamp.timestamp)

# generating training and validation sets
val_df = df.sample(frac=0.2, random_state=1337)
train_df = df.drop(val_df.index)


# split features and labels
X_train = train_df.drop(columns=["USDM"]).values
y_train = train_df["USDM"].values
X_val = val_df.drop(columns=["USDM"]).values
y_val = val_df["USDM"].values

# train model, generate classification report
model = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred, digits=3))

# predicting on entire USDM dataset
full_df = df.copy() 
X_full = full_df.drop(columns=["USDM"]).values
y_full = full_df["USDM"].values

# predict
full_df["USDM_pred"] = model.predict(X_full)
print(classification_report(y_full, full_df["USDM_pred"], digits=3))


# converting back into .nc 
# reopen USDM to get original shape
usdm_copy = xr.open_dataset("USDM05_2000_2024.nc")
usdm_base = usdm_copy["USDM"]

# make a new array filled with NaNs initially
pred_array = np.full(usdm_base.shape, np.nan)

# create a mapping from (time, lat, lon) to index for prediction values
time_index = {v: i for i, v in enumerate(usdm_base.time.values)}
lat_index = {v: i for i, v in enumerate(usdm_base.lat.values)}
lon_index = {v: i for i, v in enumerate(usdm_base.lon.values)}

# convert full_df back to datetime
full_df["time"] = pd.to_datetime(df["time"], unit="s")
# assign predicted values into the correct locations
for _, row in full_df.iterrows():
    t = time_index.get(np.datetime64(row["time"], 'ns'))
    lat = lat_index.get(row["lat"])
    lon = lon_index.get(row["lon"])
    if t is not None and lat is not None and lon is not None:
        pred_array[t, lat, lon] = row["USDM_pred"]


# create a new xarray DataArray for predictions
pred_da = xr.DataArray(
    pred_array,
    coords=usdm_base.coords,
    dims=usdm_base.dims,
    name="USDM"
)

# create dataset and save
pred_ds = xr.Dataset({"USDM": pred_da})
pred_ds.to_netcdf("USDM_predictions.nc")
