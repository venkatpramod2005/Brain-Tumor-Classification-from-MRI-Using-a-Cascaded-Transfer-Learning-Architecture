# 🚀 Deployment Guide - Brain Tumor Detection System

Complete guide for deploying the Streamlit web application locally and to the cloud (FREE).

---

## 📋 Table of Contents

1. [Local Deployment](#local-deployment)
2. [Cloud Deployment (Streamlit Cloud)](#cloud-deployment)
3. [Docker Deployment](#docker-deployment-optional)
4. [Troubleshooting](#troubleshooting)
5. [Performance Optimization](#performance-optimization)

---

## 🏠 Local Deployment

### Prerequisites

- **Python**: 3.10 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 2GB free space
- **OS**: Windows, Linux, or macOS

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd "C:\Users\venkat\Downloads\Clinically-Aware Multi-Stage Brain Tumor Intelligence System"

# Install all required packages
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Check Python version
python --version

# Check if Streamlit is installed
streamlit --version

# Check if TensorFlow is installed
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

Expected output:
```
Python 3.10.x or higher
Streamlit 1.30.x or higher
TensorFlow 2.15.x or higher
```

### Step 3: Run the Application

```bash
# Start Streamlit server
streamlit run app.py
```

The app will automatically open in your default browser at:
```
http://localhost:8501
```

### Step 4: Test the Application

1. **Upload a test image** from `samples/` folder
2. **Click "Analyze Image"**
3. **Verify prediction displays correctly**
4. **Try MC Dropout** (optional - enable in sidebar)

### Development Mode

For auto-reload on code changes:
```bash
streamlit run app.py --server.runOnSave true
```

---

## ☁️ Cloud Deployment

### Streamlit Cloud (FREE Forever)

Streamlit Cloud provides free hosting for public repositories.

### Prerequisites

- GitHub account
- Public GitHub repository
- `app.py` and `requirements.txt` in repo

### Step 1: Prepare Repository

```bash
# Initialize git (if not already)
git init

# Add files
git add .

# Commit
git commit -m "Brain tumor detection system - ready for deployment"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/brain-tumor-detection.git
git branch -M main
git push -u origin main
```

### Step 2: Handle Large Model Files

The model files (~90MB each) are too large for direct GitHub push.

**Option A: Git Large File Storage (LFS)**

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/*.keras"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add models/
git commit -m "Add models with LFS"
git push
```

**Option B: External Storage (Recommended for Free Tier)**

1. **Upload models to Google Drive**
   - Upload `best_model_mc.keras` to Google Drive
   - Make it publicly accessible
   - Get the shareable link

2. **Create download script**

Create `download_models.py`:
```python
import gdown
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive file ID (extract from share link)
FILE_ID = "YOUR_FILE_ID_HERE"

# Download
url = f"https://drive.google.com/uc?id={FILE_ID}"
output = f"{MODEL_DIR}/best_model_mc.keras"

gdown.download(url, output, quiet=False)
print("Model downloaded successfully!")
```

3. **Update requirements.txt**
```
gdown>=4.7.1
```

4. **Modify app.py to download on first run**
```python
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.info("Downloading model... (first time only)")
        import subprocess
        subprocess.run(["python", "download_models.py"])
    
    model = keras.models.load_model(MODEL_PATH)
    return model, None
```

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with GitHub

2. **Create New App**
   - Click "New app"
   - Select your repository
   - Branch: `main`
   - Main file path: `app.py`

3. **Advanced Settings** (Optional)
   ```yaml
   [server]
   maxUploadSize = 5  # Max file upload size (MB)
   
   [theme]
   primaryColor = "#4A90E2"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F8F9FA"
   textColor = "#2C3E50"
   ```

   **Important: Python Version**
   - Open **Advanced settings** in Streamlit Cloud deployment.
   - Set **Python version = 3.12**.
   - Save settings before clicking **Deploy**.

4. **Deploy**
   - Click "Deploy"
   - Wait for build (2-5 minutes first time)
   - App will be live at: `https://YOUR-APP-NAME.streamlit.app`

### Step 4: Verify Deployment

1. Visit your app URL
2. Test image upload
3. Verify predictions work
4. Test MC Dropout (if enabled)

### Step 5: Share Your App

**Public Sharing:**
```
https://YOUR-APP-NAME.streamlit.app
```

**Embed in Website:**
```html
<iframe 
  src="https://YOUR-APP-NAME.streamlit.app/?embedded=true"
  height="600" 
  style="width:100%;border:none;">
</iframe>
```

**QR Code:**
Generate QR code for easy mobile access:
- Use https://qr-code-generator.com/
- Input your app URL
- Download and share

---

## 🐳 Docker Deployment (Optional)

For more control over deployment environment.

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t brain-tumor-detection .

# Run container
docker run -p 8501:8501 brain-tumor-detection
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models:ro
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## 🔧 Troubleshooting

### Issue: Model File Not Found

**Error:**
```
Error loading model: No such file or directory
```

**Solution:**
1. Verify `models/best_model_mc.keras` exists
2. Check file permissions
3. If using external storage, ensure download script ran

### Issue: TensorFlow GPU Warning

**Warning:**
```
No GPU found. Running on CPU.
```

**Solution:**
- This is normal and expected
- CPU mode works fine for inference
- Each prediction takes ~1-2 seconds

### Issue: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Port Already in Use

**Error:**
```
Address already in use: Port 8501
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8501 | xargs kill -9
```

### Issue: Slow First Load

**Symptom:** App takes 30+ seconds to load first time

**Solution:**
- This is normal for model loading
- Subsequent loads use cache (instant)
- Consider adding loading message

### Issue: Memory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
1. Close other applications
2. Reduce MC Dropout passes
3. Process one image at a time
4. Upgrade RAM if possible

---

## ⚡ Performance Optimization

### Caching

Streamlit caching is already implemented:
```python
@st.cache_resource
def load_model():
    # Model loaded once, cached forever
    pass
```

### Reduce Model Size

**Option 1: Quantization**
```python
import tensorflow as tf

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save (50-70% smaller)
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Option 2: Pruning**
- Remove unnecessary weights
- Can reduce size by 30-50%
- Minimal accuracy loss

### Faster Predictions

1. **Disable MC Dropout by default**
   ```python
   use_mc_dropout = st.checkbox("Use MC Dropout", value=False)
   ```

2. **Reduce MC passes**
   ```python
   DEFAULT_MC_PASSES = 10  # Instead of 20
   ```

3. **Use GPU** (if available)
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       # Enable memory growth
       tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

### Optimize Visualizations

1. **Lower DPI for web**
   ```python
   fig.savefig('plot.png', dpi=150)  # Instead of 300
   ```

2. **Lazy loading**
   ```python
   with st.expander("Show detailed plots"):
       # Only generate when expanded
       st.plotly_chart(create_plot())
   ```

---

## 📊 Monitoring & Analytics

### Add Google Analytics (Optional)

In `app.py`, add to header:
```python
st.markdown("""
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-YOUR-ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-YOUR-ID');
</script>
""", unsafe_allow_html=True)
```

### Log Predictions

```python
import datetime
import json

def log_prediction(image_name, result):
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'image': image_name,
        'prediction': CLASS_NAMES[result['predicted_class']],
        'confidence': result['confidence'],
        'method': result['method']
    }
    
    with open('predictions.log', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
```

---

## 🔐 Security Considerations

### For Public Deployment

1. **Rate Limiting**
   - Streamlit Cloud has built-in rate limiting
   - Consider adding custom rate limiter if needed

2. **Input Validation**
   ```python
   # Validate file size
   if uploaded_file.size > 5 * 1024 * 1024:  # 5MB
       st.error("File too large. Max 5MB.")
       st.stop()
   
   # Validate file type
   if uploaded_file.type not in ['image/jpeg', 'image/png']:
       st.error("Invalid file type. Use JPG or PNG.")
       st.stop()
   ```

3. **HTTPS**
   - Streamlit Cloud provides HTTPS by default
   - Ensure all external resources use HTTPS

---

## 📈 Scaling

### Horizontal Scaling

For high traffic, consider:
1. **Load balancer** - Distribute requests
2. **Multiple instances** - Run parallel apps
3. **CDN** - Cache static assets

### Vertical Scaling

For better performance:
1. **More RAM** - Handle more concurrent users
2. **Better CPU** - Faster predictions
3. **Add GPU** - 10x faster inference

---

## ✅ Pre-Deployment Checklist

Before deploying to production:

- [ ] Test with multiple images
- [ ] Verify all sample images work
- [ ] Test MC Dropout functionality
- [ ] Check mobile responsiveness
- [ ] Verify model file accessible
- [ ] Test error handling
- [ ] Review security settings
- [ ] Add analytics (optional)
- [ ] Update documentation
- [ ] Create backup of models

---

## 🎯 Deployment URLs

After deployment, update README with:

```markdown
## 🌐 Live Demo

**Try it now:** https://your-app.streamlit.app

**Features:**
- Upload MRI images
- Get instant predictions
- Interactive visualizations
- MC Dropout uncertainty

**Note:** First load may take 10-20 seconds (model loading).
```

---

## 📞 Support

### Streamlit Cloud Issues
- [Streamlit Community Forum](https://discuss.streamlit.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### GitHub Issues
- Report bugs in your repository's Issues tab
- Provide error messages and steps to reproduce

---

## 🎉 Success!

Once deployed, you'll have:
- ✅ Live web application
- ✅ Shareable URL
- ✅ Free hosting (Streamlit Cloud)
- ✅ Automatic HTTPS
- ✅ Auto-deployments on git push

**Share your app and help advance medical AI research! 🚀**

---

**Deployment Guide | Streamlit Cloud | Docker | Performance**
