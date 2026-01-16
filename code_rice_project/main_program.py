import pandas as pd

url = "https://raw.githubusercontent.com/D6200763-SUT/SUT_Rice_Project_2024/refs/heads/main/Import_Dataset/file_data.csv"
#df = pd.read_csv(url)
df = pd.read_csv(url, parse_dates=["date"])
df = df.sort_values("date")
print(df.head())
print(df.shape)
station = df["address"]

print(station)