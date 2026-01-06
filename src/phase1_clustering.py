from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

SELECTED_SENSORS = [
    "sensor_2","sensor_3","sensor_4","sensor_7","sensor_8","sensor_9",
    "sensor_11","sensor_12","sensor_13","sensor_15","sensor_17",
    "sensor_20","sensor_21"
]

STAGE_MAPPING = {
    0: 'Moderately Degraded',
    1: 'Normal',
    2: 'Failure',
    3: 'Slightly Degraded',
    4: 'Critical'
}

def perform_degradation_clustering(df):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[SELECTED_SENSORS])

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(data_scaled)
    df['degradation_stage'] = df['cluster'].map(STAGE_MAPPING)
    df.drop(columns=['cluster'], inplace=True)
    return df
