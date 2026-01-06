import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_time_to_next_stage(df):
    df = df.sort_values(by=['unit', 'time']).reset_index(drop=True)
    df['time_to_next_stage'] = np.nan

    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit]
        for idx in unit_df.index:
            current_stage = df.loc[idx, 'degradation_stage']
            future = unit_df.loc[idx + 1:]
            next_rows = future[future['degradation_stage'] > current_stage]
            if not next_rows.empty:
                df.loc[idx, 'time_to_next_stage'] = (
                    next_rows.iloc[0]['time'] - df.loc[idx, 'time']
                )

    return df.dropna(subset=['time_to_next_stage'])


def train_regressor(df, sensor_cols, save_path="models/"):
    X = df[sensor_cols]
    y = df['time_to_next_stage']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))
    print("R²:", r2_score(y_test, preds))

    joblib.dump(model, save_path + "regressor.pkl")
    joblib.dump(scaler, save_path + "regressor_scaler.pkl")

    return model, scaler
