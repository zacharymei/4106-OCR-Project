# 4106-OCR-Project


## BayesOCR.py
### Parameters
```pixel_difference```
Threshold for comparing similarity between pixels. 
Make it large for ambigious image    
    
```scale```
Scaling factor for resizing images in comparison. 
***Low Scale can produce large amount of samples and require exceed memory
and may break the program to exceed maximum digit handled by Python***    
    
```output_file```
~~File output to Excel~~    
    
```print_mode```
Print the result one by one and show the current recognized character image    
### Usage
Run
```
python BayesOCR.py "your_image" 
```
It uses the default dataset in "BayesDataset" folder
If you would like to change the dataset folder, run
```
python BayesOCR.py "your_image" "dataset_folder"
```
If you would like to sit back and wait for the final result only for the whole paragraph (it takes long)    
```
python BayesOCR.py "your_image" false
```
or 
```
python BayesOCR.py "your_image" "dataset_folder" false
```

## segmentation.py
### Parameters
```tsRow```
Threshold for row segment. Make it large if the segmented does not cover enough height of row    
    
```tsChar```
Threshold for character segment. Make it large if the segmented does not cover the whole character    
    
```python
function seg(input)
input = image to be segmented
return list of segmentaed image in numpy Matrix
```
Segments images in paragraph to each single character   
character images are stored in Output folder    

### Usage
Import
```python
import segmentation
segmentation.seg(input)
```
To test the output   
```
python segmentation.py "your_image"
```
To run and disable outputing files
```
python segmentation.py "your_image" false
```
