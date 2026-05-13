# คู่มือรันโมเดลแบบเปลี่ยนไฟล์นำเข้าได้

เอกสารนี้ใช้สำหรับรันโปรแกรมฝึกสอนโมเดลโดยเปลี่ยนไฟล์ `.npz` และโฟลเดอร์ผลลัพธ์ได้ง่าย  
แนวทางหลักที่แนะนำคือใช้สคริปต์กลางชื่อ

```bash
./run_train_auto.sh <ไฟล์ .npz> <โฟลเดอร์ output>
```

---

## 1. หลักการใช้งาน

รูปแบบคำสั่งหลักคือ

```bash
./run_train_auto.sh <ไฟล์ .npz> <โฟลเดอร์ output>
```

โดยมีความหมายดังนี้

| ส่วนของคำสั่ง | ความหมาย |
|---|---|
| `./run_train_auto.sh` | ไฟล์สคริปต์กลางสำหรับรันโมเดล |
| `<ไฟล์ .npz>` | ไฟล์ sequence ที่เตรียมข้อมูลไว้แล้ว |
| `<โฟลเดอร์ output>` | โฟลเดอร์สำหรับเก็บผลการฝึกสอนโมเดล |

ตัวอย่างโครงสร้างคำสั่ง

```bash
./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

---

## 2. สิ่งที่สคริปต์จะทำให้อัตโนมัติ

เมื่อรันคำสั่ง `run_train_auto.sh` สคริปต์จะทำงานต่อเนื่องดังนี้

1. ตรวจสอบว่าไฟล์ `.npz` มีอยู่จริง
2. สร้างโฟลเดอร์ output และ logs
3. ฝึกสอนโมเดล LSTM
4. ฝึกสอนโมเดล CNN-LSTM
5. ฝึกสอนโมเดล Transformer
6. สรุปผลการทดลองด้วย `summarize_runs.py`
7. บันทึก log แยกตามโมเดล

---

## 3. ตัวอย่างการรันแต่ละชุดข้อมูล

### 3.1 รันชุด window 30 horizon 1

```bash
./run_train_auto.sh \
  out_feature_sets_w30_h1/core/sequences_window30_h1.npz \
  out_train_w30_h1
```

เหมาะสำหรับการพยากรณ์ระยะสั้น เช่น พยากรณ์ล่วงหน้า 1 วัน

---

### 3.2 รันชุด window 60 horizon 7

```bash
./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

เหมาะสำหรับการพยากรณ์ล่วงหน้า 7 วัน

---

### 3.3 รันชุด window 90 horizon 14

```bash
./run_train_auto.sh \
  out_feature_sets_w90_h14/core/sequences_window90_h14.npz \
  out_train_w90_h14
```

เหมาะสำหรับการพยากรณ์ล่วงหน้า 14 วัน

---

## 4. การให้สิทธิ์ไฟล์สคริปต์ก่อนรัน

ก่อนใช้งานครั้งแรก ให้ใช้คำสั่งนี้

```bash
chmod +x run_train_auto.sh
```

จากนั้นจึงรันคำสั่งฝึกสอนโมเดลได้

---

## 5. การรันแบบปิด Terminal ได้

ถ้าต้องการให้โปรแกรมรันต่อ แม้ปิด Terminal หรือ SSH หลุด ให้ใช้ `nohup`

ตัวอย่างสำหรับ window 60 horizon 7

```bash
nohup ./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7 \
  > train_w60_h7.log 2>&1 &
```

---

## 6. การดูสถานะระหว่างรัน

ดู log หลักแบบ real-time

```bash
tail -f train_w60_h7.log
```

ดูว่า Python ยังทำงานอยู่หรือไม่

```bash
ps aux | grep python
```

---

## 7. การดู log แยกตามโมเดล

หลังจากรันแล้ว สคริปต์จะสร้างโฟลเดอร์ logs เช่น

```text
out_train_w60_h7/logs/
```

ภายในจะมี log แยกตามขั้นตอน เช่น

```text
01_lstm_core.log
02_cnn_lstm_core.log
03_transformer_core.log
04_summary.log
```

สามารถดู log ของแต่ละโมเดลได้ เช่น

```bash
tail -f out_train_w60_h7/logs/01_lstm_core.log
```

หรือ

```bash
tail -f out_train_w60_h7/logs/03_transformer_core.log
```

---

## 8. โครงสร้างผลลัพธ์ที่ได้

เมื่อรันสำเร็จ จะได้โครงสร้างประมาณนี้

```text
out_train_w60_h7/
  lstm_core/
  cnn_lstm_core/
  transformer_core/
  summary/
  logs/
    01_lstm_core.log
    02_cnn_lstm_core.log
    03_transformer_core.log
    04_summary.log
```

---

## 9. การรันหลายชุดข้อมูล

ถ้ามีไฟล์ข้อมูลหลายชุด สามารถรันทีละชุดด้วยคำสั่งเดิมได้ เช่น

```bash
code/run_train_auto.sh \
  out_feature_sets_w30_h1/core/sequences_window30_h1.npz \
  out_train_w30_h1
```

```bash
code/run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

```bash
code/run_train_auto.sh \
  out_feature_sets_w90_h14/core/sequences_window90_h14.npz \
  out_train_w90_h14
```

---

## 10. การรันชุดใหม่ในอนาคต

ถ้ามีไฟล์ใหม่ เช่น

```text
out_feature_sets_w120_h30/core/sequences_window120_h30.npz
```

สามารถรันได้ทันทีด้วยรูปแบบเดิม

```bash
./run_train_auto.sh \
  out_feature_sets_w120_h30/core/sequences_window120_h30.npz \
  out_train_w120_h30
```

ไม่จำเป็นต้องแก้ไขโค้ดฝึกสอนโมเดลใหม่

---

## 11. คำสั่งหยุดการรัน

ถ้าต้องการหยุด Python ทั้งหมดที่กำลังรันงาน train อยู่

```bash
pkill -f "train_"
```

หรือถ้าต้องการหยุดเฉพาะ process ให้ตรวจสอบ PID ก่อน

```bash
ps aux | grep python
```

จากนั้นใช้คำสั่ง

```bash
kill PID
```

ตัวอย่าง

```bash
kill 12345
```

---

## 12. คำสั่งที่แนะนำให้ใช้จริง

ให้จำรูปแบบนี้เป็นหลัก

```bash
./run_train_auto.sh <ไฟล์ .npz> <โฟลเดอร์ output>
```

ตัวอย่างที่ใช้บ่อย

```bash
./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

แนวทางนี้ช่วยให้เปลี่ยนชุดข้อมูลได้ง่าย และทำให้ผลลัพธ์ของแต่ละการทดลองแยกกันชัดเจน
