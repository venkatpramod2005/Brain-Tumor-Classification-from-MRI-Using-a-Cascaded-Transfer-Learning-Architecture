# -*- coding: utf-8 -*-
"""
Monte Carlo Dropout - Uncertainty Estimation Module
Detects dropout layers and enables MC Dropout for robust uncertainty quantification
"""

import sys
import io
# Force UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
TEST_DIR = DATASET_DIR / "Testing"

# Model paths
BINARY_MODEL_PATH = MODELS_DIR / "best_model_binary_ResNet50_20260331_202827.keras"
MC_MODEL_PATH = MODELS_DIR / "best_model_mc.keras"

# MC Dropout settings
DEFAULT_N_PASSES = 20
VARIANCE_THRESHOLDS = {
    'low': 0.01,      # High confidence
    'medium': 0.05    # Medium confidence (>0.05 = low confidence)
}

print("=" * 80)
print("Monte Carlo Dropout - Detection & Integration Module")
print("=" * 80)


# ==================== PHASE 1: DROPOUT DETECTION ====================

def detect_dropout_layers(model):
    """
    Detect all dropout layers in the model.
    
    Returns:
        List of dictionaries with dropout layer info
    """
    dropout_layers = []
    
    for idx, layer in enumerate(model.layers):
        layer_name = layer.name.lower()
        layer_type = type(layer).__name__
        
        # Check if it's a dropout layer
        if 'dropout' in layer_name or 'dropout' in layer_type.lower():
            dropout_info = {
                'index': idx,
                'name': layer.name,
                'type': layer_type,
                'rate': getattr(layer, 'rate', None),
                'trainable': layer.trainable
            }
            dropout_layers.append(dropout_info)
    
    return dropout_layers


def print_model_summary(model, model_name):
    """Print summary of model architecture."""
    print(f"\n {model_name} Architecture Summary")
    print("-" * 80)
    print(f"  Total Layers: {len(model.layers)}")
    print(f"  Input Shape: {model.input_shape}")
    print(f"  Output Shape: {model.output_shape}")
    print(f"  Trainable Params: {model.count_params():,}")


def analyze_model_dropout(model_path, model_name):
    """
    Comprehensive dropout analysis for a model.
    
    Returns:
        Dictionary with detection results
    """
    print(f"\n{'=' * 80}")
    print(f"Analyzing: {model_name}")
    print("=" * 80)
    
    # Load model
    print(f"Loading model: {model_path.name}")
    model = keras.models.load_model(model_path)
    print("OK: Model loaded successfully")
    
    # Print summary
    print_model_summary(model, model_name)
    
    # Detect dropout layers
    print(f"\nScanning for Dropout Layers...")
    dropout_layers = detect_dropout_layers(model)
    
    if dropout_layers:
        print(f"Found {len(dropout_layers)} Dropout Layer(s):")
        for layer_info in dropout_layers:
            rate = layer_info['rate'] if layer_info['rate'] is not None else 'N/A'
            print(f"  • Layer {layer_info['index']}: {layer_info['name']}")
            print(f"    Type: {layer_info['type']}, Rate: {rate}, Trainable: {layer_info['trainable']}")
    else:
        print("No Dropout Layers Found")
    
    results = {
        'model_name': model_name,
        'model_path': str(model_path),
        'total_layers': len(model.layers),
        'dropout_layers': dropout_layers,
        'has_dropout': len(dropout_layers) > 0
    }
    
    return model, results


# ==================== PHASE 2: STOCHASTIC BEHAVIOR TESTING ====================

def load_sample_image():
    """Load a random sample image for testing."""
    import random
    from tensorflow.keras.preprocessing import image as keras_image
    
    # Get a random tumor image
    tumor_dirs = ['glioma', 'meningioma', 'pituitary']
    tumor_class = random.choice(tumor_dirs)
    class_dir = TEST_DIR / tumor_class
    
    image_files = list(class_dir.glob("*.jpg"))
    if not image_files:
        image_files = list(class_dir.glob("*.png"))
    
    if not image_files:
        raise ValueError(f"No images found in {class_dir}")
    
    sample_path = random.choice(image_files)
    
    # Load and preprocess
    img = keras_image.load_img(sample_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    
    # Ensure RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    
    # Apply ResNet50 preprocessing
    img_array = tf.cast(img_array, tf.float32)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)
    
    print(f"  Sample image loaded: {sample_path.name} ({tumor_class})")
    return img_batch, tumor_class


def test_stochastic_behavior(model, n_tests=10):
    """
    Test if dropout is active during inference by running multiple predictions.
    
    Returns:
        is_stochastic: bool
        variance: float
        predictions: array of predictions
    """
    print(f"\nTesting Stochastic Behavior ({n_tests} passes)...")
    
    # Load sample image
    sample_image, sample_class = load_sample_image()
    
    # Run multiple predictions
    predictions = []
    for i in range(n_tests):
        pred = model.predict(sample_image, verbose=0)
        predictions.append(pred[0])
        if (i + 1) % 5 == 0:
            print(f"  Pass {i + 1}/{n_tests} complete...")
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    mean_variance = np.mean(variance)
    max_variance = np.max(variance)
    
    # Determine if stochastic (variance > threshold)
    is_stochastic = mean_variance > 1e-6
    
    # Print results
    print(f"\nStochastic Behavior Test Results:")
    print(f"  Mean Variance: {mean_variance:.6f}")
    print(f"  Max Variance: {max_variance:.6f}")
    print(f"  Prediction Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    if is_stochastic:
        print(f"  STOCHASTIC - Dropout is active during inference")
    else:
        print(f"  DETERMINISTIC - Dropout is NOT active (all predictions identical)")
    
    return is_stochastic, mean_variance, predictions


# ==================== PHASE 3: DECISION LOGIC ====================

def analyze_dropout_case(detection_results, stochastic_results):
    """
    Analyze detection and stochastic test results to determine case.
    
    Cases:
        1: Dropout exists AND active - Use as-is
        2: Dropout exists BUT not active - Enable dropout
        3: No dropout layers - Fallback to entropy
    
    Returns:
        case: int (1, 2, or 3)
        recommendation: str
        action_needed: str
    """
    print(f"\n{'=' * 80}")
    print("Dropout Analysis & Recommendation")
    print("=" * 80)
    
    has_dropout = detection_results['has_dropout']
    is_stochastic = stochastic_results[0]
    
    if has_dropout and is_stochastic:
        case = 1
        recommendation = "CASE 1: Dropout exists AND is active"
        action = "Use existing dropout for MC Dropout uncertainty estimation"
        status = "optimal"
    elif has_dropout and not is_stochastic:
        case = 2
        recommendation = "CASE 2: Dropout exists BUT not active"
        action = "Enable dropout during inference for MC Dropout"
        status = "requires_activation"
    else:
        case = 3
        recommendation = "CASE 3: No dropout layers found"
        action = "Fallback to entropy-based uncertainty (MC Dropout not available)"
        status = "fallback"
    
    print(f"\n{recommendation}")
    print(f"  Has Dropout: {'Yes' if has_dropout else 'No'}")
    print(f"  Is Stochastic: {'Yes' if is_stochastic else 'No'}")
    print(f"\n  Action: {action}")
    
    results = {
        'case': case,
        'recommendation': recommendation,
        'action': action,
        'status': status,
        'has_dropout': has_dropout,
        'is_stochastic': is_stochastic
    }
    
    return results


# ==================== MAIN DETECTION WORKFLOW ====================

def run_detection_analysis():
    """Run complete detection and analysis for both models."""
    print("\n" + "=" * 80)
    print("Starting MC Dropout Detection & Analysis")
    print("=" * 80)
    
    results = {}
    
    # Analyze Multi-Class Model
    mc_model, mc_detection = analyze_model_dropout(MC_MODEL_PATH, "Multi-Class Model")
    mc_stochastic = test_stochastic_behavior(mc_model, n_tests=10)
    mc_analysis = analyze_dropout_case(mc_detection, mc_stochastic)
    
    results['multiclass'] = {
        'detection': mc_detection,
        'stochastic_test': {
            'is_stochastic': mc_stochastic[0],
            'mean_variance': float(mc_stochastic[1]),
            'predictions_shape': mc_stochastic[2].shape
        },
        'analysis': mc_analysis
    }
    
    # Analyze Binary Model
    binary_model, binary_detection = analyze_model_dropout(BINARY_MODEL_PATH, "Binary Model")
    binary_stochastic = test_stochastic_behavior(binary_model, n_tests=10)
    binary_analysis = analyze_dropout_case(binary_detection, binary_stochastic)
    
    results['binary'] = {
        'detection': binary_detection,
        'stochastic_test': {
            'is_stochastic': binary_stochastic[0],
            'mean_variance': float(binary_stochastic[1]),
            'predictions_shape': binary_stochastic[2].shape
        },
        'analysis': binary_analysis
    }
    
    # Save results
    save_detection_report(results)
    
    return results


def save_detection_report(results):
    """Save detection report as JSON."""
    report_path = BASE_DIR / "mc_dropout_detection_report.json"
    
    # Convert to JSON-serializable format
    serializable_results = {}
    for model_key, model_results in results.items():
        serializable_results[model_key] = {
            'detection': {
                'model_name': model_results['detection']['model_name'],
                'has_dropout': bool(model_results['detection']['has_dropout']),
                'dropout_count': len(model_results['detection']['dropout_layers']),
                'dropout_layers': [
                    {
                        'name': layer['name'],
                        'type': layer['type'],
                        'rate': float(layer['rate']) if layer['rate'] is not None else None
                    }
                    for layer in model_results['detection']['dropout_layers']
                ]
            },
            'stochastic_test': {
                'is_stochastic': bool(model_results['stochastic_test']['is_stochastic']),
                'mean_variance': float(model_results['stochastic_test']['mean_variance'])
            },
            'analysis': {
                'case': int(model_results['analysis']['case']),
                'recommendation': str(model_results['analysis']['recommendation']),
                'action': str(model_results['analysis']['action']),
                'status': str(model_results['analysis']['status']),
                'has_dropout': bool(model_results['analysis']['has_dropout']),
                'is_stochastic': bool(model_results['analysis']['is_stochastic'])
            }
        }
    
    # Add timestamp
    serializable_results['timestamp'] = datetime.now().isoformat()
    serializable_results['summary'] = {
        'multiclass_case': serializable_results['multiclass']['analysis']['case'],
        'binary_case': serializable_results['binary']['analysis']['case'],
        'mc_dropout_available': serializable_results['multiclass']['analysis']['has_dropout'],
        'requires_activation': serializable_results['multiclass']['analysis']['status'] == 'requires_activation'
    }
    
    with open(report_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nDetection report saved: {report_path}")


if __name__ == "__main__":
    results = run_detection_analysis()
    
    print("\n" + "=" * 80)
    print("MC Dropout Detection & Analysis Complete!")
    print("=" * 80)


# ==================== PHASE 4: ENABLE DROPOUT DURING INFERENCE ====================

def enable_dropout_at_inference(model):
    """
    Enable dropout layers during inference.
    Since setting layer.training doesn't work with functional API models,
    we'll use a wrapper that passes training=True to the model's call method.
    
    Returns:
        Wrapper function that enables dropout
    """
    print("\nEnabling Dropout for Inference...")
    
    # Count dropout layers
    dropout_count = 0
    for layer in model.layers:
        layer_name = layer.name.lower()
        if 'dropout' in layer_name:
            dropout_count += 1
            print(f"  Found dropout layer: {layer.name}")
    
    print(f"  Total dropout layers: {dropout_count}")
    print("  Using training=True in model.call() for stochastic forward passes")
    
    return model  # Return model as-is, we'll use training=True in predict calls


def verify_dropout_activation(model, sample_image, n_tests=5):
    """
    Verify that dropout is now active by using training=True.
    
    Returns:
        is_active: bool
        variance: float
    """
    print("\nVerifying Dropout Activation with training=True...")
    
    predictions = []
    for i in range(n_tests):
        # Use model() with training=True instead of predict()
        pred = model(sample_image, training=True).numpy()
        predictions.append(pred[0])
    
    predictions = np.array(predictions)
    variance = np.var(predictions, axis=0)
    mean_variance = np.mean(variance)
    
    is_active = mean_variance > 1e-6
    
    if is_active:
        print(f"  SUCCESS: Dropout is now active (variance: {mean_variance:.6f})")
    else:
        print(f"  WARNING: Dropout still not active (variance: {mean_variance:.6f})")
    
    return is_active, mean_variance


# ==================== PHASE 5: MC DROPOUT PREDICTION ====================

def mc_dropout_predict(model, image, n_passes=20, verbose=False):
    """
    Perform MC Dropout prediction with N stochastic forward passes.
    Uses model(input, training=True) to enable dropout during inference.
    
    Args:
        model: Keras model with dropout layers
        image: Preprocessed image (batch of 1)
        n_passes: Number of forward passes (default: 20)
        verbose: Print progress
    
    Returns:
        Dictionary with prediction results and uncertainty metrics
    """
    if verbose:
        print(f"\nPerforming MC Dropout ({n_passes} passes)...")
    
    predictions = []
    
    for i in range(n_passes):
        # Use model() with training=True to enable dropout
        pred = model(image, training=True).numpy()
        predictions.append(pred[0])
        
        if verbose and (i + 1) % 5 == 0:
            print(f"  Pass {i + 1}/{n_passes} complete")
    
    predictions = np.array(predictions)
    
    # Compute statistics
    mean_pred = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    std_dev = np.std(predictions, axis=0)
    
    # Overall uncertainty metrics
    mean_variance = np.mean(variance)
    max_variance = np.max(variance)
    
    # Predicted class and confidence
    predicted_class = np.argmax(mean_pred)
    confidence = np.max(mean_pred)
    
    # Entropy of mean prediction
    epsilon = 1e-10
    entropy_val = -np.sum(mean_pred * np.log2(mean_pred + epsilon))
    
    results = {
        'mean_prediction': mean_pred,
        'predicted_class': int(predicted_class),
        'confidence': float(confidence),
        'variance': variance,
        'mean_variance': float(mean_variance),
        'max_variance': float(max_variance),
        'std_dev': std_dev,
        'entropy': float(entropy_val),
        'all_predictions': predictions,
        'n_passes': n_passes
    }
    
    if verbose:
        print(f"  Mean variance: {mean_variance:.6f}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"  Predicted class: {predicted_class}")
    
    return results


def classify_uncertainty(mean_variance, thresholds=None):
    """
    Classify prediction uncertainty based on variance.
    
    Args:
        mean_variance: Average variance across classes
        thresholds: Dict with 'low' and 'medium' thresholds
    
    Returns:
        uncertainty_level: str
        color_code: str
        message: str
    """
    if thresholds is None:
        thresholds = VARIANCE_THRESHOLDS
    
    if mean_variance < thresholds['low']:
        return 'High Confidence', 'success', 'Prediction is reliable'
    elif mean_variance < thresholds['medium']:
        return 'Medium Confidence', 'warning', 'Moderate uncertainty detected'
    else:
        return 'Low Confidence', 'error', 'High uncertainty - Review Recommended'


# ==================== PHASE 6: BATCH PROCESSING ====================

def mc_dropout_predict_batch(model, images, labels, n_passes=20):
    """
    Perform MC Dropout prediction on a batch of images.
    
    Returns:
        List of prediction results
    """
    print(f"\nProcessing {len(images)} images with MC Dropout ({n_passes} passes each)...")
    
    results = []
    
    for i, (img_path, true_label) in enumerate(zip(images, labels)):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(images)}")
        
        # Load and preprocess
        img = load_and_preprocess_image(img_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # MC Dropout prediction
        pred_results = mc_dropout_predict(model, img_batch, n_passes=n_passes, verbose=False)
        
        # Add metadata
        pred_results['image_path'] = str(img_path)
        pred_results['true_label'] = true_label
        
        # Classify uncertainty
        uncertainty_level, color, message = classify_uncertainty(pred_results['mean_variance'])
        pred_results['uncertainty_level'] = uncertainty_level
        pred_results['uncertainty_color'] = color
        pred_results['uncertainty_message'] = message
        
        results.append(pred_results)
    
    print(f"  Completed: {len(results)} predictions")
    
    return results


def load_and_preprocess_image(img_path):
    """Load and preprocess a single image for prediction."""
    from tensorflow.keras.preprocessing import image as keras_image
    
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    
    # Ensure RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    
    # Apply ResNet50 preprocessing
    img_array = tf.cast(img_array, tf.float32)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    
    return img_array


# ==================== MAIN TESTING FUNCTION ====================

def test_mc_dropout_pipeline():
    """Test the complete MC Dropout pipeline."""
    print("\n" + "=" * 80)
    print("Testing MC Dropout Pipeline")
    print("=" * 80)
    
    # Load model
    print("\nLoading multi-class model...")
    model = keras.models.load_model(MC_MODEL_PATH)
    print("Model loaded")
    
    # Enable dropout
    model = enable_dropout_at_inference(model)
    
    # Load sample image
    sample_image, sample_class = load_sample_image()
    
    # Verify activation
    is_active, variance = verify_dropout_activation(model, sample_image, n_tests=5)
    
    if not is_active:
        print("\nWARNING: Dropout could not be activated!")
        print("Falling back to entropy-based uncertainty estimation")
        return None
    
    # Perform MC Dropout prediction
    print("\n" + "-" * 80)
    results = mc_dropout_predict(model, sample_image, n_passes=20, verbose=True)
    
    # Classify uncertainty
    uncertainty_level, color, message = classify_uncertainty(results['mean_variance'])
    
    print("\n" + "-" * 80)
    print("MC Dropout Prediction Results:")
    print(f"  Predicted Class: {results['predicted_class']}")
    print(f"  Confidence: {results['confidence']:.4f}")
    print(f"  Mean Variance: {results['mean_variance']:.6f}")
    print(f"  Entropy: {results['entropy']:.4f}")
    print(f"  Uncertainty Level: {uncertainty_level}")
    print(f"  Message: {message}")
    print("-" * 80)
    
    return results


if __name__ == "__main__":
    # Run detection if not already done
    import os
    if not os.path.exists(BASE_DIR / "mc_dropout_detection_report.json"):
        results = run_detection_analysis()
    
    # Test MC Dropout pipeline
    test_results = test_mc_dropout_pipeline()
    
    if test_results:
        print("\n" + "=" * 80)
        print("MC Dropout Pipeline Test SUCCESSFUL!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("MC Dropout not available - use entropy-based uncertainty")
        print("=" * 80)

