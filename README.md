Follow me on LinkedIn : [click](https://www.linkedin.com/in/jitendarkumarsaini25/)


# Age and Gender Detection #
 

 
This project uses OpenCV and deep learning models to detect faces in images or webcam feeds and predict the age and gender of the detected individuals. The model utilizes pre-trained deep learning frameworks for face detection, age prediction, and gender classification.

## Table of Contents ##
- Features
- Requirements
- Installation
- Usage
- Command Line Arguments
- License
  
## Features ##
- Real-time face detection using a webcam or an image file.
- Predicts gender and age of detected faces.
- Customizable rectangle and text colors for display.
  
## Requirements ##
- Python 3.x
- OpenCV
- NumPy
  
You can install the required packages using pip:

```
pip install opencv-python opencv-python-headless numpy
```

## Installation ##

1. Clone the repository:

```
git clone https://github.com/jitendra205saini/webcam_gender_finder_python.git
cd webcam_gender_finder_python
```
2. Download the pre-trained models for face detection, age estimation, and gender classification, and place them in the specified directories:

- Face Detection Model:
   - opencv_face_detector.pbtxt
   - opencv_face_detector_uint8.pb
- Age Model:
   - age_deploy.prototxt
   - age_net.caffemodel
- Gender Model:
   - gender_deploy.prototxt
   - gender_net.caffemodel
Ensure the model paths in the code match the locations of these files.

## Usage ##
Run the script from the command line:

```
python gender.py --image 0
```
You can replace ``` 0 ``` with the path to an image file if you want to analyze a static image instead of using the webcam.

## Command Line Arguments ##
- ``` --image ```: Path to the image file or ``` 0 ```to use the webcam (default: ``` 0 ```).
- ``` --frame_color ```: Color of the rectangle around detected faces in B,G,R format (default:``` 0,255,0 ```).
- ``` --text_color ```: Color of the age and gender text in B,G,R format (default: ``` 154,14,234 ```).

### Example ###
```
python gender.py --image 0 --frame_color 255,0,0 --text_color 0,255,255
```
