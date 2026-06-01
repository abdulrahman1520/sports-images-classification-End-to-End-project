
# 🏅 Sports Image Classification

A deep learning project that classifies sports images into 7 categories using a custom CNN model trained with extensive hyperparameter tuning. Deployed as a production-ready Flask web application.

---

## 📌 Sports Categories

`Badminton` · `Cricket` · `Karate` · `Soccer` · `Swimming` · `Tennis` · `Wrestling`

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| Final Validation Accuracy | **96.35%** |
| Image Size | 64 × 64 |
| Best Batch Size | 16 |
| Best Number of Layers | 4 |
| Best Dropout Rate | 0.3 |
| Best Optimizer | Adam |
| Best Weight Decay | 0.001 |
| Learning Rate | 0.001 |
| LR Scheduler | ✅ ReduceLROnPlateau |

---

## 🧠 Model Architecture

- Custom **CNN** built with TensorFlow/Keras
- 4 convolutional blocks with **BatchNormalization** and **MaxPooling**
- Dropout regularization (rate = 0.3)
- L2 weight decay (0.001)
- Output: Softmax over 7 classes

---

## 🔬 Hyperparameter Tuning Process

Each hyperparameter was explored independently and the best value carried forward:

| Hyperparameter | Options Tested | Best |
|----------------|---------------|------|
| Data Standardization | None vs Standardized | Standardized (88.39%) |
| Batch Size | 16, 32, 64 | 16 (88.66%) |
| Num Layers | 2, 3, 4 | 4 (91.86%) |
| Dropout Rate | 0.3, 0.5, 0.7 | 0.3 (92.30%) |
| Optimizer | SGD, Adam, RMSprop | Adam (92.61%) |
| Weight Decay | 0.0, 0.0001, 0.001 | 0.001 (95.02%) |
| Learning Rate | 0.01, 0.001, 0.0001 | 0.001 (92.49%) |
| LR Scheduler | False, True | True (93.50%) |

---

## 📊 Dataset

- Source: [Sports Image Classification — Kaggle](https://www.kaggle.com/datasets)
- **Imbalanced classes** — Karate and Swimming had significantly fewer images
- Applied **data augmentation** (factor = 3) to balance classes
  - Original: 8,227 images → Augmented: 24,681 images
- Augmentation techniques: flipping, rotation, brightness adjustment

---

## 🗂️ Project Structure

```
sports-image-classification/
│
├── app.py                               # Flask web app
├── sports-images-classification.ipynb  # Full training notebook
├── model/
│   ├── sports_classifier_model.keras   # Trained model
│   └── preprocessing_params.json       # Mean, std, class names
├── static/
│   └── uploads/                        # Temp folder for uploaded images
├── templates/
│   ├── index.html                      # Upload page
│   └── result.html                     # Prediction result page
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/sports-image-classification.git
cd sports-image-classification
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask app
```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000`

---

## 💬 How to Use

1. Open the app in your browser
2. Upload a sports image (JPG, JPEG, or PNG)
3. The app returns:
   - **Predicted sport category**
   - **Confidence score**
   - **Top 3 predictions** with probabilities

---

## 📦 Dependencies

```
tensorflow
flask
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## 🚀 Future Enhancements

- Try transfer learning with MobileNetV2 or EfficientNet for higher accuracy
- Increase image size (128×128 or 224×224)
- Add more sports categories
- Deploy on cloud (Heroku / Render / AWS)
```
