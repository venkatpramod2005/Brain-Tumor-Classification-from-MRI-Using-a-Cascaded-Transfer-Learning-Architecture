# MC Dropout Quick Reference Guide

## ✅ MC Dropout IS Integrated!

### 📍 Location in App

**SIDEBAR (Left Panel)**
```
⚙️ Settings
├── Classification Mode
│   ○ Binary (Tumor Detection)
│   ○ Multi-Class (Tumor Type)
│   ● Two-Stage (Both)  ← Selected by default
│
├── ☐ Use MC Dropout  ← CHECK THIS BOX!
│   └── Number of MC passes [10────●────50]
│       (Slider appears when checked)
│
├── 📊 Model Information
├── ℹ️ About
└── 🖼️ Sample Images
```

### 🎯 How to Enable

1. **Find the checkbox** in sidebar under "Settings"
2. **Check** "Use MC Dropout"
3. **Adjust** passes (default: 20)
4. **Upload** an image
5. **Click** "🔍 Analyze Image"

### 📊 What You'll See

**WITHOUT MC Dropout** (checkbox unchecked):
- Fast prediction (~1 second)
- Method: "Standard"
- Entropy-based uncertainty
- Single confidence value

**WITH MC Dropout** (checkbox checked):
- Progress bar: "MC Dropout: Pass 1/20..."
- Slower (~2-5 seconds)
- Method: "MC Dropout"
- Variance-based uncertainty
- Additional visualization: "MC Dropout Prediction Distribution"
- In detailed metrics: "Mean Variance" and "MC Passes"

### 🧪 Test It

1. **Without MC Dropout:**
   - Uncheck the box
   - Upload `sample_glioma.jpg`
   - Click Analyze
   - Note the speed (~1s)
   - Check results: Method = "Standard"

2. **With MC Dropout:**
   - CHECK the box ✓
   - Keep same image
   - Click Analyze again
   - Watch progress bar (Pass 1/20, 2/20, etc.)
   - Takes longer (~2-3s)
   - Check results: Method = "MC Dropout"
   - Scroll down: See box plot distribution

### 🔍 Troubleshooting

**Don't see the checkbox?**
- Refresh browser (F5 or Ctrl+R)
- Make sure sidebar is open (click ▶ arrow if collapsed)
- Check browser console for errors (F12)

**Checkbox is there but grayed out?**
- Should work fine, just old browser styling

**Not seeing difference when checked?**
- Make sure to click "Analyze Image" AFTER checking
- Watch for progress text during prediction
- Check "Detailed Metrics" expander for "MC Passes: 20"

### 💡 When to Use MC Dropout

**Use MC Dropout when:**
- ✓ Accuracy is critical
- ✓ You need reliable uncertainty estimates
- ✓ Time permits (2-5 seconds acceptable)
- ✓ Making clinical decisions
- ✓ Want to see prediction variability

**Skip MC Dropout when:**
- ✓ Speed is priority
- ✓ Batch processing many images
- ✓ Quick screening
- ✓ Don't need detailed uncertainty

### 📈 Understanding MC Dropout Output

**Mean Variance** (key metric):
- < 0.01: High confidence ✓
- 0.01 - 0.05: Medium confidence ⚠
- > 0.05: Low confidence ❌

**Box Plot** (MC Dropout Prediction Distribution):
- Shows all 20 predictions
- Tight box = consistent predictions (good!)
- Wide box = variable predictions (uncertain)
- Helps visualize model uncertainty

### ✅ Verification Checklist

To confirm MC Dropout is working:
- [ ] Checkbox visible in sidebar
- [ ] Slider appears when checked
- [ ] Progress bar shows during prediction
- [ ] Takes noticeably longer (2-5s vs 1s)
- [ ] Results show "Method: MC Dropout"
- [ ] Detailed metrics include "Mean Variance"
- [ ] Box plot appears (for multi-class)
- [ ] Progress text: "MC Dropout: Pass X/20"

---

**All checkboxes above should be ✓ if MC Dropout is integrated properly!**

If not seeing these, try:
1. Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
2. Clear browser cache
3. Restart Streamlit server
4. Check terminal for errors
