# Weather Dataset Analysis Report

**Total Images**: 16918
**Number of Classes**: 4
**Classes**: cloudy, rainy, snowy, sunny

## Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| cloudy | 6640 | 39.25% |
| rainy | 1828 | 10.81% |
| snowy | 1562 | 9.23% |
| sunny | 6888 | 40.71% |

**Imbalance Ratio** (max/min): 4.41
**Is Balanced**: No — consider class weights or oversampling

## Image Size Statistics

| Metric | Value |
|--------|-------|
| Min Width | 124px |
| Max Width | 5184px |
| Mean Width | 465.0px |
| Min Height | 47px |
| Max Height | 3317px |
| Mean Height | 396.8px |
| Mean Aspect Ratio | 1.241 |
| Recommended Resize | **384px** |

## Image Formats

| Format | Count |
|--------|-------|
| .jpg | 16917 |
| .jpeg | 1 |

## File Size Statistics

- Min: 0.002 MB
- Max: 5.354 MB
- Mean: 0.041 MB
- Total: 687.66 MB

## Bad / Corrupted Images

**Count**: 0

## Preprocessing Recommendations

1. Resize all images to **384×384px**
2. Use RandomResizedCrop during training for robustness to varied sizes
3. Convert all images to RGB (handle grayscale, RGBA)
4. Use conservative ColorJitter — weather depends on color information
5. Skip the identified bad images during training
6. Apply class weights or focal loss to handle class imbalance