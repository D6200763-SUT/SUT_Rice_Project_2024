import pandas as pd

#url = "https://raw.githubusercontent.com/D6200763-SUT/SUT_Rice_Project_2024/refs/heads/main/Import_Dataset/file_data.csv"
#url = "https://raw.githubusercontent.com/D6200763-SUT/SUT_Rice_Project_2024/refs/heads/main/Import_Dataset/Rice_Cultivated_Area_By_Variety_Season_Monthly_2015_2019/Monthly-production%20volume%20of%20rice%20in-season%20Rai(unit)%202015.csv"

url = "https://raw.githubusercontent.com/D6200763-SUT/SUT_Rice_Project_2024/refs/heads/main/Import_Dataset/Environmental_Features.csv"

df = pd.read_csv(url)
df = pd.read_csv(url, parse_dates=["Date"])
df = df.sort_values("Date")
print(df.head())
print(df.shape)

cols = df.columns.tolist()
print(cols)

col = "StationID"   # เปลี่ยนเป็นชื่อคอลัมน์ของคุณ

names_list = (
    df[col]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

print("จำนวนชื่อทั้งหมด =", len(names_list))
print(names_list[:])   # แสดงตัวอย่าง 30 รายการแรก


'''
#cols = df.columns.tolist()
#print(cols)
cols_new = ['date', 'address', 'latitude', 'longitude', 'day', 'month', 'year', 'mirid bug', 'mint', 'maxt', 'temp', 'dew', 'humidity', 'wspd', 'wdir', 'precip','bph']
print(cols_new)    # <-- ใส่ชื่อคอลัมน์ที่คุณต้องการ
new_df = df.loc[:, cols_new].copy()

# เปลี่ยนชื่อคอลัมน์
new_df = new_df.rename(columns={
    "date": "Date",
    "address": "StationID",
    "mirid bug": "mirid_bug",
    'mint': 'mint_temp', 
    'maxt':'maxt_temp',
    'temper':'temperature',
    'dew':'dew_point',
    'wspd':'wind_speed', 
    'wdir':'wind_direction',
    'precip':'rainfall',
    'bph':'BPH_count'
})

# บันทึกเป็นไฟล์ใหม่ (แนะนำ utf-8-sig เพื่อเปิดใน Excel ไทยได้สวย)
new_df.to_csv("Environmental_Features.csv", index=False, encoding="utf-8-sig")

print(new_df.shape)
#station = df["address"]

#print(station)
'''