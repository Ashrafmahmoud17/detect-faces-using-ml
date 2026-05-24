# 🎭 Face Recognition System
### MediaPipe + Logistic Regression | Real-Time Person Identification

---

## 📌 Overview

A real-time face recognition system that identifies specific individuals using **facial landmark detection** via MediaPipe Face Mesh and a **Logistic Regression** classifier. The system captures 468 facial landmarks (x, y coordinates) from a webcam feed and predicts the identity of the detected person in real time.

This project was built to recognize two individuals: **Ashraf** and **Mohamed**.

---

## 🧠 How It Works

```
Webcam Feed
     │
     ▼
MediaPipe Face Mesh (468 landmarks)
     │
     ▼
Extract X, Y coordinates → Feature Vector (936 features)
     │
     ▼
MinMaxScaler → Normalize features
     │
     ▼
Logistic Regression Model → Predict Identity
     │
     ▼
Display Name on Frame
```

---

## 📁 Project Structure

```
face-recognition/
│
├── create-data.ipynb       # Collect facial landmark data from webcam
├── train_model.ipynb       # Train the Logistic Regression classifier
│
├── dataset1.csv            # Generated dataset (landmarks + labels)
├── model.pkl               # Trained Logistic Regression model
├── scaler.pkl              # Fitted MinMaxScaler
│
└── saved_faces/            # Auto-saved prediction screenshots
```

---

## ⚙️ Requirements

### Python Version
```
Python 3.8+
```

### Libraries
```bash
pip install opencv-python
pip install mediapipe
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
```

---

## 🚀 Getting Started

### Step 1 — Collect Data (`create-data.ipynb`)

Run this notebook to collect facial landmark data for each person:

1. Open `create-data.ipynb`
2. Run the first section → webcam opens and captures **Ashraf's** landmarks
3. Press **`q`** when done collecting
4. Run the second section → captures **Mohamed's** landmarks
5. Press **`q`** when done
6. The dataset is automatically saved as `dataset1.csv`

> **Tip:** Collect at least **200–500 frames** per person for better accuracy. Move your head slightly during capture for variety.

---

### Step 2 — Train the Model (`train_model.ipynb`)

1. Open `train_model.ipynb`
2. Run all cells
3. The notebook will:
   - Load `dataset1.csv`
   - Split data (90% train / 10% test)
   - Apply **MinMaxScaler** normalization
   - Train a **Logistic Regression** model
   - Save `model.pkl` and `scaler.pkl`

---

### Step 3 — Run Real-Time Recognition

The last section of `create-data.ipynb` runs real-time prediction:

1. Make sure `model.pkl` and `scaler.pkl` exist
2. Run the prediction cell
3. Webcam opens — your name appears on the screen in real time
4. Press **`q`** to exit

---

## 📊 Dataset Details

| Feature | Description |
|---|---|
| Columns 0–467 | X coordinates of 468 facial landmarks |
| Columns 468–935 | Y coordinates of 468 facial landmarks |
| `output` | Label: `Ashraf` or `Mohamed` |
| Total features | 936 per frame |
| Sampling rate | Every 5 frames (skip=5) |

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Scaler | MinMaxScaler |
| Test size | 10% |
| Shuffle | True |
| Random state | 42 |
| Stratify | Yes |

---

## 💾 Auto-Save Feature

The system automatically saves a screenshot every **20 frames** when a face is detected:

```
saved_faces/
├── Ashraf_0.jpg
├── Ashraf_20.jpg
├── Mohamed_40.jpg
└── ...
```

---

## 🔧 Customization

### Add a New Person
1. In `create-data.ipynb`, duplicate the data collection block
2. Change the variable name (e.g., `sara = []`)
3. Set the label: `sara['output'] = 'Sara'`
4. Add to the concat: `final_output = pd.concat((Ashraf, Mohamed, sara), axis=0)`
5. Re-run `train_model.ipynb`

### Change Camera Source
```python
# Default webcam
cap = cv2.VideoCapture(0)

# External camera
cap = cv2.VideoCapture(1)

# Video file
cap = cv2.VideoCapture("video.mp4")
```

---

## 📈 Tips for Better Accuracy

- Collect data in the **same lighting conditions** you'll use for recognition
- Include frames with **slight head movement** (left, right, up, down)
- Collect at least **300 frames** per person
- Avoid very dark or very bright backgrounds during collection
- Keep the camera at a **consistent distance** (40–80 cm)

---

## 👤 Author

**Ashraf Mahmoud**
Computer Sciences — New Mansoura University

---

## 📄 License

This project is for educational purposes.
