# ความรู้พื้นฐานเกี่ยวกับไฟล์ `.sh` และ `.npz`

เอกสารนี้สรุปความหมาย วิธีใช้งาน และบทบาทของไฟล์ `.sh` และ `.npz` ในงานพัฒนาโปรแกรมและงานฝึกสอนโมเดล Machine Learning / Deep Learning โดยเฉพาะงานพยากรณ์ข้อมูลอนุกรมเวลา เช่น LSTM, CNN-LSTM และ Transformer

---

## 1. ควรสร้างเป็นไฟล์อะไรดี

แนะนำให้สร้างเป็นไฟล์

```text
README_file_sh_npz.md
```

หรือ

```text
docs/file_sh_npz_guide.md
```

เหตุผลที่แนะนำใช้ไฟล์ `.md` หรือ Markdown คือ

1. เปิดอ่านได้ง่ายใน VS Code
2. แสดงผลสวยใน GitHub
3. ใช้เป็นเอกสารประกอบโปรเจกต์ได้
4. แก้ไขเพิ่มเติมง่าย
5. เหมาะสำหรับเขียนคู่มือ คำสั่ง ตัวอย่างโค้ด และบันทึกความรู้
6. สามารถนำไปต่อยอดเป็น README หลักของโปรเจกต์ได้

ถ้าต้องการส่งให้นักศึกษา หรือแนบรายงานอย่างเป็นทางการ อาจแปลงจาก `.md` เป็น `.docx` หรือ `.pdf` ภายหลังได้

---

## 2. ไฟล์ `.sh` คืออะไร

ไฟล์ `.sh` คือ Shell Script

เป็นไฟล์ที่ใช้เก็บชุดคำสั่งสำหรับระบบ Linux, Ubuntu หรือ macOS เพื่อให้สามารถสั่งงานผ่าน Terminal ได้อัตโนมัติ

กล่าวอย่างง่ายคือ

```text
.sh = ไฟล์รวมคำสั่ง Terminal หลายคำสั่งไว้ในไฟล์เดียว
```

แทนที่จะพิมพ์คำสั่งยาว ๆ ทีละบรรทัดทุกครั้ง เราสามารถเขียนคำสั่งทั้งหมดไว้ในไฟล์ `.sh` แล้วรันไฟล์เดียวได้

---

## 3. ตัวอย่างไฟล์ `.sh`

ตัวอย่างชื่อไฟล์

```text
run_train_auto.sh
```

ภายในไฟล์อาจมีคำสั่ง เช่น

```bash
python code/11_train_lstm_compat.py \
  --npz out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  --out_dir out_train_w60_h7/lstm_core
```

เมื่อสร้างเป็นไฟล์ `.sh` แล้ว สามารถรันได้ด้วยคำสั่ง

```bash
./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

---

## 4. ไฟล์ `.sh` ใช้งานในรูปแบบใด

ไฟล์ `.sh` เหมาะกับงานที่ต้องรันคำสั่งซ้ำ ๆ หรือรันหลายขั้นตอนต่อเนื่อง เช่น

1. เตรียมโฟลเดอร์ output
2. ตรวจสอบไฟล์ input
3. รันโมเดล LSTM
4. รันโมเดล CNN-LSTM
5. รันโมเดล Transformer
6. สรุปผลการทดลอง
7. บันทึก log การทำงาน

ตัวอย่างการใช้งานจริง

```bash
chmod +x run_train_auto.sh
```

จากนั้นรัน

```bash
./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

---

## 5. คำสั่ง `chmod +x` คืออะไร

ก่อนรันไฟล์ `.sh` ครั้งแรก ควรใช้คำสั่ง

```bash
chmod +x run_train_auto.sh
```

ความหมายคือ ให้สิทธิ์ไฟล์นี้สามารถรันได้

ถ้าไม่ใช้คำสั่งนี้ อาจเจอ error ประมาณว่า

```text
Permission denied
```

---

## 6. ไฟล์ `.npz` คืออะไร

ไฟล์ `.npz` คือไฟล์ข้อมูลของ NumPy ในภาษา Python

ใช้สำหรับเก็บข้อมูล array หลายชุดไว้ในไฟล์เดียว เหมาะสำหรับงาน Machine Learning และ Deep Learning

กล่าวอย่างง่ายคือ

```text
.npz = ไฟล์ข้อมูลที่เตรียมพร้อมแล้วสำหรับป้อนเข้าโมเดล
```

ในงานพยากรณ์อนุกรมเวลา ไฟล์ `.npz` มักเป็นข้อมูลที่ผ่านขั้นตอน preprocessing แล้ว เช่น

1. อ่านข้อมูล CSV
2. รวมข้อมูลหลายแหล่ง
3. จัดการ missing values
4. สร้าง feature
5. ทำ normalization หรือ scaling
6. แปลงข้อมูลเป็น sliding window
7. แบ่ง train / validation / test

จากนั้นจึงบันทึกเป็นไฟล์ `.npz`

---

## 7. ตัวอย่างชื่อไฟล์ `.npz`

ตัวอย่างไฟล์

```text
sequences_window30_h1.npz
sequences_window60_h7.npz
sequences_window90_h14.npz
```

ความหมายโดยทั่วไป

| ชื่อไฟล์ | ความหมาย |
|---|---|
| `sequences_window30_h1.npz` | ใช้ข้อมูลย้อนหลัง 30 วัน เพื่อพยากรณ์ล่วงหน้า 1 วัน |
| `sequences_window60_h7.npz` | ใช้ข้อมูลย้อนหลัง 60 วัน เพื่อพยากรณ์ล่วงหน้า 7 วัน |
| `sequences_window90_h14.npz` | ใช้ข้อมูลย้อนหลัง 90 วัน เพื่อพยากรณ์ล่วงหน้า 14 วัน |

---

## 8. ภายในไฟล์ `.npz` มักมีอะไร

ไฟล์ `.npz` สำหรับฝึกโมเดลมักเก็บข้อมูล เช่น

```text
X_train
y_train
X_val
y_val
X_test
y_test
feature_names
```

ความหมายของแต่ละส่วน

| ตัวแปร | ความหมาย |
|---|---|
| `X_train` | ข้อมูล input สำหรับฝึกโมเดล |
| `y_train` | คำตอบจริงของชุด train |
| `X_val` | ข้อมูล input สำหรับตรวจสอบระหว่างฝึก |
| `y_val` | คำตอบจริงของชุด validation |
| `X_test` | ข้อมูล input สำหรับทดสอบโมเดล |
| `y_test` | คำตอบจริงของชุด test |
| `feature_names` | รายชื่อ feature ที่ใช้ในโมเดล |

---

## 9. ตัวอย่างรูปแบบข้อมูลใน `.npz`

ตัวอย่าง shape ของข้อมูล

```text
X_train shape = (42432, 60, 14)
y_train shape = (42432,)
```

แปลความหมายได้ว่า

```text
มีข้อมูลฝึกทั้งหมด 42,432 ตัวอย่าง
แต่ละตัวอย่างใช้ข้อมูลย้อนหลัง 60 วัน
แต่ละวันมี 14 features
มีคำตอบจริง 42,432 ค่า
```

สำหรับโมเดล LSTM, CNN-LSTM และ Transformer ข้อมูล input มักอยู่ในรูปแบบ

```text
(samples, timesteps, features)
```

เช่น

```text
(42432, 60, 14)
```

---

## 10. ความสัมพันธ์ของ `.sh`, `.npz`, และ `.py`

ในโปรเจกต์ Machine Learning จะมีไฟล์หลัก ๆ ดังนี้

```text
.py  = โปรแกรม Python ที่ทำงานจริง
.npz = ไฟล์ข้อมูลที่เตรียมพร้อมแล้ว
.sh  = ไฟล์คำสั่งสำหรับสั่งรันโปรแกรมอัตโนมัติ
```

เปรียบเทียบให้เข้าใจง่าย

| ไฟล์ | เปรียบเทียบ | หน้าที่ |
|---|---|---|
| `.npz` | วัตถุดิบ | ข้อมูลสำหรับป้อนเข้าโมเดล |
| `.py` | เครื่องจักร | โปรแกรมฝึกโมเดลหรือประมวลผล |
| `.sh` | ใบสั่งงาน | สั่งให้เครื่องจักรทำงานตามลำดับ |

---

## 11. ตัวอย่าง workflow ของโปรเจกต์

```text
ข้อมูลดิบ CSV
   ↓
เตรียมข้อมูลด้วย Python
   ↓
สร้างไฟล์ .npz
   ↓
ใช้ไฟล์ .sh สั่งรันโมเดล
   ↓
เรียกใช้ไฟล์ .py
   ↓
ได้ผลลัพธ์โมเดลและ summary
```

ตัวอย่างจริง

```text
out_feature_sets_w60_h7/core/sequences_window60_h7.npz
   ↓
run_train_auto.sh
   ↓
11_train_lstm_compat.py
12_train_cnn_lstm_compat.py
13_train_transformer_compat.py
summarize_runs.py
   ↓
out_train_w60_h7/
```

---

## 12. ตัวอย่างคำสั่งที่แนะนำให้ใช้จริง

รูปแบบหลัก

```bash
./run_train_auto.sh <ไฟล์ .npz> <โฟลเดอร์ output>
```

ตัวอย่าง

```bash
./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

ความหมายคือ

```text
นำไฟล์ sequences_window60_h7.npz ไปฝึกโมเดล
แล้วเก็บผลลัพธ์ทั้งหมดไว้ในโฟลเดอร์ out_train_w60_h7
```

---

## 13. ตัวอย่างการรันหลายชุดข้อมูล

### window 30 horizon 1

```bash
./run_train_auto.sh \
  out_feature_sets_w30_h1/core/sequences_window30_h1.npz \
  out_train_w30_h1
```

### window 60 horizon 7

```bash
./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

### window 90 horizon 14

```bash
./run_train_auto.sh \
  out_feature_sets_w90_h14/core/sequences_window90_h14.npz \
  out_train_w90_h14
```

---

## 14. ทำไมต้องแยก input และ output

การแยก input และ output ทำให้จัดการผลการทดลองได้ง่าย เช่น

```text
Input:
out_feature_sets_w60_h7/core/sequences_window60_h7.npz

Output:
out_train_w60_h7
```

ข้อดีคือ

1. รู้ว่าผลลัพธ์มาจากข้อมูลชุดใด
2. เปรียบเทียบผลแต่ละ window/horizon ได้ง่าย
3. ลดโอกาสเขียนทับผลการทดลองเดิม
4. จัดทำตารางเปรียบเทียบผลวิจัยได้สะดวก
5. เหมาะสำหรับการทำซ้ำการทดลองในงานวิจัย

---

## 15. การนำไปใช้เป็นแนวทางพัฒนาโปรแกรม

แนวคิดนี้สามารถนำไปพัฒนาโปรแกรมให้เป็นระบบมากขึ้นได้ เช่น

1. เปลี่ยน input ได้โดยไม่แก้โค้ด Python
2. กำหนด output แต่ละการทดลองได้ชัดเจน
3. สร้าง log แยกตามรอบการทดลอง
4. นำไปต่อยอดเป็นระบบ experiment manager
5. นำไปใช้กับหลายโมเดลได้
6. นำไปใช้กับหลายชุดข้อมูลได้
7. เหมาะสำหรับงานวิจัยที่ต้องทดลองซ้ำหลายเงื่อนไข

---

## 16. ข้อดีของการใช้ `.sh` ร่วมกับ `.npz`

| ประเด็น | ข้อดี |
|---|---|
| ความสะดวก | รันคำสั่งยาว ๆ ได้ด้วยคำสั่งเดียว |
| ความถูกต้อง | ลดการพิมพ์ path ผิด |
| การทำซ้ำ | ทดลองซ้ำได้ง่าย |
| การจัดการผลลัพธ์ | แยก output ตามชุดข้อมูล |
| งานวิจัย | เหมาะกับการเปรียบเทียบโมเดล |
| การสอน | ใช้เป็นตัวอย่าง workflow ให้นักศึกษาได้ |

---

## 17. ข้อควรระวัง

1. ต้องตรวจสอบว่า path ของไฟล์ `.npz` ถูกต้อง
2. ต้องตรวจสอบว่าไฟล์ `.sh` มีสิทธิ์รัน
3. อย่าใช้ output folder เดิม หากไม่ต้องการให้ผลลัพธ์ถูกเขียนทับ
4. ควรตั้งชื่อ output ให้สัมพันธ์กับ input
5. ควรเก็บ log ทุกครั้ง เพื่อใช้ตรวจสอบปัญหาภายหลัง

---

## 18. สรุปสั้นที่สุด

```text
.sh  คือ ไฟล์คำสั่งอัตโนมัติ
.npz คือ ไฟล์ข้อมูลที่เตรียมพร้อมสำหรับฝึกโมเดล
.py  คือ โปรแกรม Python ที่ทำงานจริง
```

การใช้งานจริงในโปรเจกต์

```bash
./run_train_auto.sh \
  out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  out_train_w60_h7
```

แปลว่า

```text
ใช้ข้อมูล window60 horizon7 เป็น input
รันโมเดลทั้งหมด
บันทึกผลลัพธ์ไว้ที่ out_train_w60_h7
```

---

## 19. โครงสร้างไฟล์ที่แนะนำในโปรเจกต์

```text
SUT_Rice_Project_2024/
  code/
    11_train_lstm_compat.py
    12_train_cnn_lstm_compat.py
    13_train_transformer_compat.py
    summarize_runs.py

  out_feature_sets_w30_h1/
    core/
      sequences_window30_h1.npz

  out_feature_sets_w60_h7/
    core/
      sequences_window60_h7.npz

  out_feature_sets_w90_h14/
    core/
      sequences_window90_h14.npz

  out_train_w30_h1/
  out_train_w60_h7/
  out_train_w90_h14/

  run_train_auto.sh

  docs/
    README_file_sh_npz.md
```

---

## 20. แนวทางตั้งชื่อไฟล์เอกสาร

แนะนำใช้ชื่อใดชื่อหนึ่งต่อไปนี้

```text
README_file_sh_npz.md
```

หรือ

```text
docs/file_sh_npz_guide.md
```

หรือถ้าใช้เป็นคู่มือในโปรเจกต์วิจัย

```text
docs/experiment_file_structure_guide.md
```

ชื่อที่แนะนำที่สุดคือ

```text
docs/file_sh_npz_guide.md
```

เพราะสั้น ชัดเจน และเหมาะกับการเก็บเป็นเอกสารประกอบโปรเจกต์
