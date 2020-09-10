import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    # #############################################################################
    # # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    # #############################################################################

    # Compute the horizontal and vertical derivatives
    sobel_kernel = 3
    image_bw_Ix = cv2.Sobel(image, cv2.CV_64FC1, 1, 0, ksize = sobel_kernel)
    image_bw_Iy = cv2.Sobel(image, cv2.CV_64FC1, 0, 1, ksize = sobel_kernel)
    

    # Compute three images corresponding to outer product of gradients
    cutoff_frequency = 16
    kernel = cv2.getGaussianKernel(ksize = cutoff_frequency, sigma = cutoff_frequency/32, ktype = cv2.CV_64F)
    kernel = np.dot(kernel, kernel.T)
    image_bw_Ixx = cv2.filter2D(image_bw_Ix * image_bw_Ix, -1, kernel)
    image_bw_Iyy = cv2.filter2D(image_bw_Iy * image_bw_Iy, -1, kernel)
    image_bw_Ixy = cv2.filter2D(image_bw_Ix * image_bw_Iy, -1, kernel)

    # Compute scalar interest measure
    alpha = 0.04
    image_r = image_bw_Ixx * image_bw_Iyy - image_bw_Ixy ** 2 - alpha * (image_bw_Ixx + image_bw_Iyy) ** 2
    # image_r = image_r / np.max(image_r)

    # Find Harris corners
    threshold = 0.1
    (image_row, image_col) = np.where(image_r > threshold)
    x, y = image_col, image_row
    thresh_image_r = image_r[image_row, image_col]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################
    numPoints = 3000
    image_interestVal = thresh_image_r
    sorted_image_interestVal = image_interestVal[np.flip(np.argsort(image_interestVal))]
    sorted_image_row = image_row[np.flip(np.argsort(image_interestVal))]
    sorted_image_col = image_col[np.flip(np.argsort(image_interestVal))]

    distance = []
    distance.append(999999)
    for i in range(1, len(sorted_image_row)):
        dist = [0]
        xi, yi = sorted_image_col[i], sorted_image_row[i]
        for j in range(0, i):
            xj, yj = sorted_image_col[j], sorted_image_row[j]
            if sorted_image_interestVal[i] > 1.1 * sorted_image_interestVal[j]:
                r = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                dist.append(r)
        minDist = np.amin(np.array(dist))
        distance.append(minDist)
    distance = np.array(distance)

    x = sorted_image_col[np.flip(np.argsort(distance))]
    y = sorted_image_row[np.flip(np.argsort(distance))]

    i = np.where(x > feature_width//2 - 1)
    x, y = x[i], y[i]
    i = np.where(y > feature_width//2 - 1)
    x, y = x[i], y[i]

    i = np.where(x < np.shape(image)[1] - feature_width//2)
    x, y = x[i], y[i]
    i = np.where(y < np.shape(image)[0] - feature_width//2)
    x, y = x[i], y[i]

    x = x[0: numPoints]
    y = y[0: numPoints]           

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y, confidences, scales, orientations
