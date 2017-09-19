## CarND-Advanced-Lane-Lines-WriteUp
---

**Advanced Lane Finding Project**

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

[image1]: ./projectwritup_photoes/undistor_comparsion.png "Undistorted"
[image2]: ./projectwritup_photoes/filter.png "Thresholded"
[image3]: ./projectwritup_photoes/draw_line.png "Detectlines"
[image4]: ./projectwritup_photoes/putback.png "Drawback line"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 95 through 104 section 4  in `Advanced-Lane-Lines.ipynb`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

    gradx = abs_sobel_thresh(image, orient='x',  thresh=(2, 100))
    grady = abs_sobel_thresh(image, orient='y',  thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=15, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.0))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1) & (hls_binary ==1 ) | 
              (mag_binary == 1) & (dir_binary == 1) & (hls_binary ==1))  | 
             (gradx == 1) & (grady == 1) & (mag_binary == 1)
            ] = 1    
    

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 22 through 36 in 2nd code cell  of the file `Advanced-Lane-Lines.ipynb`    The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

    M = get_perspective_transform()
    warped = cv2.warpPerspective(img, M, img_size)   

```python
    src = np.float32([[203,700],[1076,720],[557,474],[723,474]])
    img_size = (img.shape[1], img.shape[0])              
    original_X = 320
    original_Y = 360
    extend_X = 640
    extend_Y = 360
    dst = np.float32([[original_X, original_Y+extend_Y],[original_X +extend_X, original_Y+extend_Y], 
                                     [original_X, original_Y], 
                                     [original_X + extend_X, original_Y]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The filter steps really take a lot of time to adjust to get proper lines detect.At beginning, in the shadow image, the 2 line curve differently, later it turn out the righ lane has very little pixes and very close pixes to fit, so I have to retouch the threshhold to make the right lane has more pix which cause a lot of noise and later choose the combined as above.

Then I use line histgraom to find the lines area, and slid window to find the pixel and adjust center of window when necessary on assumed line, then I use np.polyfit interpose the line.

I do that in function  fine_lane(binary_warped):

![alt text][image3]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

do that in function def fine_lane(binary_warped):

    ploty = np.linspace(0, 719, num=720)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0]) 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in plot_lane   Here is an plot_lane
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img = np.dstack((img, img, img))*255

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.figure()
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

![alt text][image4]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most the pipeline work but under shadown photo which fail to reconganize the line, and I have turn the old clear picture into a lot of noise pic so it can still catch up small gradicent at earlier phase, and later filter out correctly.