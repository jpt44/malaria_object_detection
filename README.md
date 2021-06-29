# Malaria Multi-Label Classification and Object Detection

Goal: identify types of malaira-infected cells that exist in an image

Strategies:
1. (Classification Only) Feed each image through a CNN and let it identify the type of infected cell(s) the image contains without finding its location in the image, i.e. no bounding boxes. 

   | Feature Extractor |  Overall Accuracy (%)   |  Overall Precision (%)     | Overall Recall (%) | Overall F-Score (%)|
   | :------------- | :----------: | :-----------: | :-----------: | :-----------: |
   |ResNet50V2|87.4|94.3|66.6|78.1|
   |InceptionV3|87.1|90.1|69.2|78.3|
   |Xception|87.8|94.7|67.5|78.8|
   
   |Feature Extractor|Class Accuracy (%)|Class Precision (%)|Class Recall (%)|Class F-score (%)|
   | :------------------------- | :----------------- | :----------- | :----------- | :----------- |
   |ResNet50V2|100,91.5,87.4,88.5,80,76.9 |100,NaN,NaN,NaN,NaN,80.8|100,0,0,0,0,69.9|100,NaN,NaN,NaN,NaN,75|
   |InceptionV3|100,89.7,87.9,88.5,81.3,74.9 |100,37,56.2,NaN,55.8,86.8|100,30.3,18.4,0,30.8,58|100,33.3,27.7,NaN,39.7,69.5|
   |Xception|100, 91.5,87.4,88.5,79.7,79.5 |100,NaN,NaN,NaN,45.5,85.1|100,0,0,0,6.4,71|100,NaN,NaN,NaN,11.2,77.4|


2. (Object Detection Problem) Feed each image through a CNN and let it identify the type of infected cell(s) the image contains and find it’s location, i.e. find bounding boxes
   
   |Feature Extractor|mAP (IoU=0.5) (%)|AR@100 (%)|AR@10 (%)|AR@1 (%)|
   | :------------- | :----------: | :-----------: | :-----------: | :-----------: |
   |SSD ResNet50 V1 FPN 640x640 (RetinaNet50)|64.1|74.2|63|41.2|
   |EfficientDet D2 768x768|34.7|51.9|41.4|25.9|
   |SSD MobileNet V1 FPN 640x640|66.3|72.8|61.7|41.1|
   
   |Feature Extractor|Overall Accuracy (%)|Overall Precision (%)|Overall Recall (%)|Overall F-score (%)|
   | :------------- | :----------: | :-----------: | :-----------: | :-----------: |
   |SSD RetinaNet50 640x640|91.98|69.67|70.64|70.15|
   |EfficientDet D2 768x768|88.3|NaN|37.49|Nan|
   |SSD MobileNet V1 FPN 640x640|91.3|67|68.4|67.7|

   |Feature Extractor|Class Accuracy (%)|Class Precision (%)|Class Recall (%)|Class F-score (%)|
   | :------------------------- | :----------------- | :----------- | :----------- | :----------- |
   |SSD RetinaNet50 640x640|100,96.5,89.4,87,91.1,87.9|100,73.7,51.6,37.7,78.7,76.4|100,84.8,32.7,57.8,61.5,87|100,78.9,40,45.6,69.1,81.4|
   |EfficientDet D2 768x768|100,92.2,87.5,87.3,82.6,80|100,NaN,34.6,25.9,17.9,63.8|100,0,18.4,15.6,17.9,73.1|100,NaN,24,19.4,24.8,68.1|
   |SSD MobileNet V1 FPN 640x640|100,96.8,90.3,89.4,86.8,84.7|100,72.1,56.8,45.5,50,67.7|100,93.9,42.9,33.3,54.5,90.2|100,81.6,48.8,38.5,54.5,77.3|


 
