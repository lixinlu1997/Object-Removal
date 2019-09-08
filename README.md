README.md for 442 Project by Group 19

# Introduction
This project offers an efficient and effective method for filling the removed regions in an image.

# Environment
The code can be compile based on eta library (eta library installation info can be found in:https://github.com/voxel51/eta ) and the other library we will use is:

- Keras 2.1.5
- tensorflow 1.6.0
- PIL
- skimage.io
- scipy
- imageio

# Usage
- Before start, first you need to download a pre-trained YOLOv3 weights, by entering:
  ```bash
  wget https://pjreddie.com/media/files/yolov3.weights
  ```
  Store this .weights file in the root directory.

- Then put the image you want to process in '/image' directory. If you have had a mask for your image, put it in '/mask' directory.

- Run 'python main.py'

- It will let you choose the mode:
  1. enter '1' if you don't have a mask for your image and the object you want to remove
  2. enter '2' if you have the maks for your image and the object you want to remove

- Input the filename.

- It will show the patchsize it has found.

- If you choose the first mode (entered '1'), an image wil show. In the image, the objects have been detected will be marked with rectangle. 
  Each object will be labeled an index on the left top corner. Then input the index(es) of the object(s) you want to remove, split with ','
  For example ('0' or '1,2' or '0,1,3'). 

- If you choose the second mode (entered '2'), you will be asked to input the filename of the mask image.

- Then the process of filling will be shown dynamically.

- After all the regions have been filled, the final image and the gif for process will be saved in '/output' directory.

- It will ask you whether you want to process next image. (y/n)

# Detailed information
We have include 4 sample images in '/image' directory. 2 of them have no prepared masks and 2 of them have prepared masks.
For 'sample1.jpg', we choose the people with index 1 after the objects have been detected by yolo.
For 'sample2.png', we choose the poeple with index 0 after the objects have been detected by yolo.
For 'sample3.jpg' and 'sample4.jpg', the correspinding masks have been placed in the '/mask' directory, named 'sample3_mask.jpg' and 'sample4_mask.jpg'.

# Example Demonstration
Here is the removal process of an example,
![](example.gif)
More example is pre-included in ./output directory

# keras-yolo3
We have used yolo3 to help detect the object in the image. The pre-trained weights have been downloaded to use directly.

https://pjreddie.com/darknet/yolo/
 
