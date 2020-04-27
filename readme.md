## Color Segmentation for Stop Sign Detection

### Task Description:

Train a probabilistic color model from image data and use it to segment unseen images, detect
stop signs in an image, and draw bounding boxes around them. Given a set of training images, you should
hand-label examples of different colors by selecting appropriate pixels. From these examples, you should
build color classifiers for several colors (e.g., red, yellow, brown, etc.) and finally a stop sign detector. You
should then use your algorithm to obtain the bounding box of a detected stop sign in the image frame on
new test images.

### Code organisation:

```
mask_create.py :        Used to create the mask.
logistic_regres.py :    Code for supervised learning.
stop_sign_detector.py : Code for detecting the stop sign
weights_MLE.npy :       Contains the final weights of the trained model.
```
### Results:

![Image Description](master/results/bbox/30.png)
