# 🧠 Brain Tumor Detection System

A beautiful, modern Streamlit web application for brain tumor classification using deep learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![License](https://img.shields.io/badge/License-Research-green.svg)

---

## 🎯 Overview

This application uses a pre-trained ResNet50 deep learning model to classify brain MRI scans into three tumor types:

- 🔴 **Glioma** - Tumors arising from glial cells
- 🔵 **Meningioma** - Tumors arising from the meninges
- 🟢 **Pituitary** - Tumors in the pituitary gland

## ✨ Features

### 🖼️ Image Upload
- Drag-and-drop interface
- Support for JPG and PNG formats
- Real-time image preview

### 🧠 AI Analysis
- Fast prediction (<1 second)
- Multi-class tumor classification
- Confidence scoring (0-100%)

### 📊 Interactive Visualizations
- Confidence gauge meter
- Probability distribution chart
- Color-coded uncertainty badges

### 💾 Export Results
- Download predictions as JSON
- Timestamped reports
- Complete probability distribution

### 🎨 Beautiful UI
- Modern gradient design
- Responsive layout
- Custom styling with CSS
- Medical-themed color scheme

---

## 📸 Screenshots

### Main Interface
The app features a clean two-column layout:
- **Left**: Image upload and preview
- **Right**: Analysis results and visualizations

### Results Display
- Large, bold tumor type prediction
- Visual confidence gauge
- Interactive probability chart
- Color-coded uncertainty indicator

---

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

4. **Open in browser**
```
http://localhost:8501
```

### Using Sample Images
Sample MRI images are included in the `samples/` folder:
- `glioma_sample.jpg`
- `meningioma_sample.jpg`
- `pituitary_sample.jpg`

---

## 📁 Project Structure

```
brain-tumor-detection/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/
│   └── best_model_mc.keras     # Pre-trained ResNet50 model
├── samples/
│   ├── glioma_sample.jpg
│   ├── meningioma_sample.jpg
│   └── pituitary_sample.jpg
├── dataset/                    # Training/Testing data (optional)
├── evaluation_results/         # Model evaluation metrics
└── visualizations/             # Generated charts
```

---

## 🎨 UI Design

### Color Scheme
| Element | Color | Usage |
|---------|-------|-------|
| Primary | `#667eea` | Headers, buttons |
| Glioma | `#f5576c` | Glioma predictions |
| Meningioma | `#4facfe` | Meningioma predictions |
| Pituitary | `#38f9d7` | Pituitary predictions |
| High Confidence | `#2ecc71` | Confidence ≥80% |
| Medium Confidence | `#f39c12` | Confidence 50-80% |
| Low Confidence | `#e74c3c` | Confidence <50% |

### Typography
- **Font**: Inter (Google Fonts)
- **Headings**: Bold, large
- **Body**: Regular, readable

---

## 🔧 Technical Details

### Model Architecture
- **Base**: ResNet50 (pre-trained on ImageNet)
- **Input Shape**: 224 × 224 × 3
- **Output**: 3 classes (softmax)
- **Preprocessing**: ResNet50 standard preprocessing

### Performance Metrics
- **Accuracy**: 84.11%
- **Binary ROC AUC**: 0.9882
- **PR AUC**: 0.9946

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 85.91% | 83.33% | 84.60% |
| Meningioma | 85.10% | 70.92% | 77.36% |
| Pituitary | 81.94% | 98.33% | 89.39% |

### Uncertainty Estimation
Uses entropy-based uncertainty calculation:
```
H = -Σ(p_i × log₂(p_i))
```
- **High Confidence**: Confidence ≥ 80%
- **Medium Confidence**: 50% ≤ Confidence < 80%
- **Low Confidence**: Confidence < 50%

---

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare Repository
1. Create a GitHub repository
2. Push all files including:
   - `app.py`
   - `requirements.txt`
   - `models/best_model_mc.keras`
   - `samples/` (optional)

### Step 2: Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"

### Step 3: Configure (Optional)
- Add custom subdomain
- Set Python version in `runtime.txt`
- Configure secrets if needed

### Deployment Tips
- Model file must be under 200MB
- Use Git LFS for large files
- Clear cache if deployment fails

---

## 📋 Requirements

### Python Version
- Python 3.10 or higher

### Core Dependencies
```
streamlit>=1.30.0
tensorflow>=2.15.0
keras>=3.0.0
numpy>=1.24.0
plotly>=5.18.0
Pillow>=10.0.0
```

### Full Dependencies
See `requirements.txt` for complete list.

---

## 🔬 How It Works

### 1. Image Upload
User uploads a brain MRI image (JPG/PNG format).

### 2. Preprocessing
```python
# Resize to 224x224
image = image.resize((224, 224))

# Apply ResNet50 preprocessing
img_array = preprocess_input(img_array)
```

### 3. Prediction
```python
# Generate probabilities
probabilities = model.predict(img_array)

# Get predicted class
predicted_class = TUMOR_CLASSES[np.argmax(probabilities)]
```

### 4. Uncertainty Calculation
```python
# Calculate entropy
entropy = -np.sum(probs * np.log2(probs))
normalized_entropy = entropy / np.log2(3)
```

### 5. Display Results
- Tumor type with color-coded card
- Confidence gauge (Plotly)
- Probability bar chart
- Uncertainty badge

---

## ⚠️ Disclaimer

**This system is for research and educational purposes only.**

- Not a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals
- Do not use for clinical decision-making

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is for research and educational purposes.
See LICENSE file for details.

---

## 🙏 Acknowledgments

- ResNet50 architecture by Microsoft Research
- Brain tumor dataset contributors
- Streamlit team for the amazing framework
- TensorFlow/Keras community

---

## 📞 Contact

For questions or issues:
- Open a GitHub issue
- Email: your.email@example.com

---

**Built with ❤️ using Streamlit, TensorFlow, and Plotly**

© 2026 Brain Tumor Detection System
