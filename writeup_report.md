#**Self-Driving Car Engineer Nanodegree**

##**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chess_undistort.png "Chess Undistorted"
[image2]: ./output_images/test1_undistort.png "Test1 Undistorted"
[image3]: ./output_images/rgb_channels.png "RGB Channels"
[image4]: ./output_images/hls_channels.png "HLS Channels"
[image5]: ./output_images/sobelx_threshold.png "Threshold Sobelx"
[image6]: ./output_images/magnitude_threshold.png "Threshold Magnitude"
[image7]: ./output_images/direction_threshold.png "Threshold Direction"
[image8]: ./output_images/combined_threshold.png "Threshold Combined"
[image9]: ./output_images/pipeline_threshold.png "Threshold Pipeline"

[image10]: ./output_images/perspective_straight.png "Perspective Straight"
[image11]: ./output_images/perspective_curve.png "Perspective Curve"
[image12]: ./output_images/perspective_pipeline.png "Perspective Pipeline"

[image13]: ./output_images/fitline_perspective.png "Fit lines"
[image14]: ./output_images/fitline_perspective_second.png "Fit lines Second frame"
[image15]: ./output_images/curvature_position.png "Curvature & Position"

[video1]: ./output_images/project_video_result.gif "Video Result"

---
###Files

My project includes the following files:
* advanced_lane_finding.ipyng containing the scripts to detect lane boundaries
* writeup_report.md summarizing the results

---

###Camera Calibration

The code for this step is contained in the code cells 2 and 3 of the IPython notebook "advanced_lane_finding.ipynb". The code is tested in cells 5 to 7 of the same notebook file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

---

###Pipeline (single images)

Using the `objpoints` and `imgpoints` computed earlier we can now apply distortion correction on one of the test images (code cell 7 from IPython notebook) with the following result:
![alt text][image2]

---

###Thresholded binary image

Gradient threshold methods presented in the lesson, Sobel operator, Magniture of the Gradient and Direction of the Gradient are implemented in code cell 10 of "advanced_lane_finding.ipynb".
Example output of each of this methods is presented below:

![alt text][image5]
![alt text][image6]
![alt text][image7]

As we can see none of this methods (as implemented in the notebook) can't be used by itself to succesfully identify the line lanes but we can also try to combine all / some of them:

```
# Apply each of the thresholding functions
gradx = abs_sobel_threshold(test1_image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_threshold(test1_image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_threshold(test1_image, sobel_kernel=ksize, thresh=(30, 100))
dir_binary = dir_threshold(test1_image, sobel_kernel=15, thresh=(0.7, 1.1))

test1_combined = np.zeros_like(mag_binary)
test1_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```

getting the the following output:

![alt text][image8]

As we can see the result is still not the expected one. Of course, we can still improve this by trying to use different threshold values or to combine the methods in a different way, but one of the reason why those methods failed, more or less, is because it was first converted to grayscale so we didn't take advantage of the color information.

####RGB vs HLS

To better understand which image channels can be useful in this task I used one of the test image to visualize each channel separately (code cells 8, 9 from "advanced_lane_finding.ipynb"):

![alt text][image3]

`R` color channel seems to be the best in identifying both yellow and white lanes.

![alt text][image4]

In this case, `S` color channel looks like the best option to use.


####Final Pipeline

The final pipeline I came up with (code cell 18 from notebook) use a combination of SobelX with threshold values (55, 100) on `R` channel and it is also take advantage of `S` channel using a threshold of (110, 255).

Here's an example of my threshold pipeline output:

![][image9]

---

###Perspective Transformation

The code for my perspective transform includes a function called `perspective_transform()` and one called `warp_perspective()`, which appears in the code cell nr. 22 of the IPython notebook. The `perspective_transform()` function takes as inputs an image (img) and use some hardcoded values of source (src) destination (dst) points. I chose the hardcode the source and destination points in the following manner:

| Source        | Destination   |
|:-------------:|:-------------:|
|  682, 445     | 938, 0        |
| 1062, 690     | 938, 720      |
|  260, 690     | 367, 720      |
|  600, 445     | 367, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![][image10]

And the result of perspective transformed of a curve:

![][image11]

And also the result when perspective transformation is applied on a binary version:

![][image12]

---

###Detect lane pixels and fit to find the lane boundary

The code to detect lane pixels and line fit includes a function called `fit_lines` which appears in the code cell nr. 29 of the IPython notebook.
The function gets as a parameter a binary warped image and, as explained in lessons, it first try to guess where the lane lines are by creating a histogram of the bottom half of the image and extracts two different peak areas.
Once we have the peaks for left and right line the next step is to use use sliding windows to determine the line direction from the bottom to the top of the image.
Here is an example of lane lines determined using the `fit_lines` function:

![][image13]

Once we have the lane lines for the first frame, for the second one we can save some time and skip the full sliding windows caculation by using the old lines as a starting point for searching the new once (code cell nr. 32 of the IPython notebook).
Here is an example output of using `fit_lines_second_frame` function:

![][image14]

As we can see the lane lines are not perfectly parallel and this can happen because of brithness, color lane and other factors. To fix this issue on video pipeline we assign weights to each frame and then those frames where the lines are not well determine influence less on the final result.

---

###Curvature of the lane and vehicle position with respect to center

The code to determine curvature of the lane includes a function called `curvature` which appears in the code cell nr. 34 of the IPython notebook.
First I had to convert the space from pixels to real world dimensions and then I calculated the polynomial fit in that space. The final result is the average of the two lines.

For vehicle position with respect to center, the function `vehicle_position` (code cell nr. 36 of IPython notebook) is determine by calculating the offset of the lane center from the center of the image (converted from pixels to meters).

---

###Result

The function `draw_full_pipeline` (code cell nr. 38 of IPython notebook) applies all the above steps on each image or video frame.
Here's an example of the final result applied on a two individual images:

![alt text][image15]

---

###Pipeline (video)

The code cell nr. 45 of IPython notebook process a video by going through each video frame and applies, like in single example example, all the above steps generating the following result (converted to gif):

![alt text][video1]

Here's a [link to my video result](./output_videos/project_video_result.mp4)

---

###Discussion

This was very challenging project. Now I have a better understanding about computer vision and how can be used in tasks like detecting lane lines.

My approch waas to apply all the techniques discussed in the lessons and try to figure it out which one works best in my case.

The final pipeline is using Sobelx and saturation threshold and it seems to make a good job on the project video but on the challenge video was not big success.

This means that the current pipeline overfits the project video, which is true because I did all the tests and adjustments using only the test images which are part of the project video.

As an improvement for this I should try to extract same frames from the other two videos (challenge and harder challenge) to apply the same process step by step to see exacly where it fails.

Another way to improve the current pipeline can be made by checking if detected lane lines make sense (distance between them, slope, etc.). Currently I assign weights for each frame but this helps only with jitering but if the lanes are not well detected this can't help. What we can do instead is to skip same lines (and use the old once) if in same frames we are not able to correctly detect them.
