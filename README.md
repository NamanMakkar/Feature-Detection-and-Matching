# **Project 2:  Feature Detection and Matching**
<br> 
CS5670 - Intro to Computer Vision<br> 
<br> 
Project Details Page: https://www.cs.cornell.edu/courses/cs5670/2023sp/projects/pa2/
<br>
<br>

## Introduction:
The goal of feature detection and matching is to identify a pairing between a point in one image and a corresponding point in another image. These correspondences can then be used to stitch multiple images together into a panorama. In this project, we detected image features and matching pairing features as following:

- Feature detection using Harris
- Feature description (simple and MOPS)
- Feature descriptor SIFT (Scale Invariant Feature Transform) implemented in bonus directory in 'extracredit.py'
- Feature matching (SSD and ratio)

<br>

## Results:
### Input:
Yosemite 1 & Yosemite 2 
<br>
![Yosemite 1](./resources/yosemite/yosemite1.jpg)
<br> 
![Yosemite 2](./resources/yosemite/yosemite2.jpg)

<br> 

### Output: Keypoint Detection:
Yosemite 1: 1559 keypoints detected
<br>
![Yosemite 1](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Yosemite1%20-%20Keypoint%20Detection.jpg)
<br> 
Yosemite 2: 1356 keypoints detected
<br> 
![Yosemite 2](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Yosemite2%20-%20Keypoint%20Detection.jpg)

<br>

### Output: Feature Matching
MOPS SSD: 1559 Matches
<br>
![MOPS SSD](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Yosemite%20-%20Feature%20Matching%20MOPS%20SSD%20.jpg)
<br>
MOPS Ratio: 1559 Matches 
<br>
![MOPS Ratio](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Yosemite%20-%20Feature%20Matching%20MOPS%20Ratio.jpg)
<br>


Simple SSD: 1559 Matches
<br>
![Simple SSD](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Yosemite%20-%20Feature%20Matching%20Simple%20SSD.jpg)
<br>
Simple Ratio: 1559 Matches
<br>
![Simple Ratio](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Yosemite%20-%20Feature%20Matching%20Simple%20Ratio.jpg)
<br>
<br> 

### Performance Benchmark
ROC Curves and their respective AUC values for all matcher types and tests are shown below in the figure: 
<br>
![ROC Curves & AUC values](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Benchmark%20-%20ROC%20Curves%20%26%20AUC%20values.jpg)
<br>
<br>

## Extra Credit: SIFT ROC Curves
### ROC Curves:
SIFT SSD Test ROC Curve: AUC Value 0.9841
<br>
![SIFT SSD Test ROC Curve](./results/Benchmark_SIFT_SSD_0.98417.png)
<br>
SIFT Ratio Test ROC Curve: AUC Value 0.9858
<br>
![SIFT Ratio Test ROC Curve:](./results/Benchmark_SIFT_Ratio_0.9858.png)

<br> 

### Feature Matching Results:

SIFT SSD
<br>
![SIFT SSD](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Yosemite%20-%20Feature%20Matching%20SIFT%20SSD.jpg)
<br>
SIFT Ratio
<br>
![SIFT Ratio](https://github.com/NamanMakkar/Feature-Detection-and-Matching/blob/main/results/Yosemite%20-%20Feature%20Matching%20SIFT%20Ratio%20Test.jpg)
