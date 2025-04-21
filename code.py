import pandas as pd

df = pd.read_csv("/content/train_FD001.txt", sep=" ", header=None)
# Drop empty columns at the end
df.dropna(axis=1, inplace=True)

# Add column names
df.columns = ["unit", "time"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

unit1 = df[df['unit'] == 1]
scaler = MinMaxScaler()

for sensor in ["sensor_6"]:
    normalized = scaler.fit_transform(unit1[[sensor]])
    plt.plot(unit1['time'], normalized, label=sensor)

plt.xlabel("Time (cycle)")
plt.ylabel("Normalized Sensor Value")
plt.title("Normalized Sensor Trends for Unit 1")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


