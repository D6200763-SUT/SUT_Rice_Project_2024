import pandas as pd

#url = "https://raw.githubusercontent.com/D6200763-SUT/SUT_Rice_Project_2024/refs/heads/main/Import_Dataset/file_data.csv"
url = "https://raw.githubusercontent.com/D6200763-SUT/SUT_Rice_Project_2024/refs/heads/main/Import_Dataset/Rice_Cultivated_Area_By_Variety_Season_Monthly_2015_2019/Monthly-production%20volume%20of%20rice%20in-season%20Rai(unit)%202015.csv"
df = pd.read_csv(url, encoding="utf-8-sig")
#df = pd.read_csv(url, parse_dates=["date"])
#df = df.sort_values("date")

cols = df.columns.tolist()
print(cols)

#print(df.head())
#print(df.shape)
#station = df["address"]

#print(station)