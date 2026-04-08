from PIL import Image
import numpy as np
from pathlib import Path

viz_dir = Path('C:/Users/venkat/Downloads/Clinically-Aware Multi-Stage Brain Tumor Intelligence System/visualizations')

required_files = [
    'confusion_matrix.png',
    'roc_curve.png',
    'precision_recall_curve.png',
    'class_performance.png',
    'confidence_distribution.png',
    'error_analysis.png'
]

print('=' * 80)
print('VALIDATION REPORT')
print('=' * 80)

all_valid = True
for filename in required_files:
    filepath = viz_dir / filename
    if filepath.exists():
        img = Image.open(filepath)
        arr = np.array(img)
        
        print(f'\n{filename}:')
        print(f'  Mode: {img.mode}')
        print(f'  Size: {img.size} pixels')
        print(f'  DPI: {img.info.get("dpi", "N/A")}')
        print(f'  Array shape: {arr.shape}')
        print(f'  Data type: {arr.dtype}')
        print(f'  Value range: [{arr.min()}, {arr.max()}]')
        
        if img.mode == 'RGB' and arr.dtype == np.uint8 and len(arr.shape) == 3 and arr.shape[2] == 3:
            print(f'  Status: ✅ VALID')
        else:
            print(f'  Status: ❌ INVALID')
            all_valid = False
    else:
        print(f'\n{filename}: ❌ FILE NOT FOUND')
        all_valid = False

print('\n' + '=' * 80)
if all_valid:
    print('✅ ALL VISUALIZATIONS VALIDATED SUCCESSFULLY')
else:
    print('❌ SOME VALIDATIONS FAILED')
print('=' * 80)
