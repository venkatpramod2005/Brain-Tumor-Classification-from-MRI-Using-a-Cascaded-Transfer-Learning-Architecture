"""
MC Dropout Test Script - Verify Integration
Run this to test MC Dropout functionality directly
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

print("=" * 60)
print("MC Dropout Integration Test")
print("=" * 60)

# Load model
MODEL_PATH = Path("models/best_model_mc.keras")
print(f"\n1. Loading model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("   ✓ Model loaded")

# Create dummy image
print("\n2. Creating test image...")
dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
print("   ✓ Test image created")

# Test 1: Standard prediction
print("\n3. Testing STANDARD prediction (dropout OFF)...")
pred_standard = model.predict(dummy_image, verbose=0)
print(f"   Result shape: {pred_standard.shape}")
print(f"   Method: Standard")

# Test 2: MC Dropout prediction
print("\n4. Testing MC DROPOUT prediction (dropout ON)...")
predictions = []
for i in range(5):
    pred_mc = model(dummy_image, training=True).numpy()
    predictions.append(pred_mc[0])
    print(f"   Pass {i+1}/5: {pred_mc[0]}")

predictions = np.array(predictions)
variance = np.var(predictions, axis=0)
mean_variance = np.mean(variance)

print(f"\n   Mean prediction: {np.mean(predictions, axis=0)}")
print(f"   Variance: {variance}")
print(f"   Mean Variance: {mean_variance}")
print(f"   Method: MC Dropout")

# Verify dropout is active
print("\n5. Verification:")
if mean_variance > 1e-6:
    print("   ✓ MC DROPOUT IS WORKING!")
    print(f"   ✓ Predictions vary (variance: {mean_variance:.6f})")
else:
    print("   ✗ MC Dropout NOT working")
    print("   ✗ Predictions identical (variance ≈ 0)")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)

print("\nFor MC Dropout to work in the app:")
print("1. CHECK the box: Use MC Dropout")
print("2. Upload image")
print("3. Click Analyze")
print("4. Look for: Method: MC Dropout (not Standard)")
print("5. Should see: Mean Variance metric")
print("6. Progress bar: MC Dropout: Pass X/20")
