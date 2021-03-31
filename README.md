# 4106-OCR-Project

## segmentation.py
```
tsRow = threshold for Row
```
Make it large if the segmented does not cover enough height of row
```
tsChar = threshold for Character
```
Make it large if the segmented does not cover the whole character
```python
function seg(input, tsRow, tsChar)
input = image to be segmented
return list of segmentaed image in numpy Matrix
```
Segments images in paragraph to each single character   
character images are stored in Output folder

