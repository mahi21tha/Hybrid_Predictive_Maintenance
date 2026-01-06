from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def add_operating_clusters(df, n_clusters=6):
    scaler = StandardScaler()
    op_scaled = scaler.fit_transform(df[['op1', 'op2', 'op3']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['op_cluster'] = kmeans.fit_predict(op_scaled)
    return df

def normalize_sensors_by_cluster(df, sensor_cols):
    df[sensor_cols] = df.groupby('op_cluster')[sensor_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df
