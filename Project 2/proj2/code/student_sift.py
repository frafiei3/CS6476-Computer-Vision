import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """


    ## UNCOMMENT THIS PART IF YOU ARE USING CHEAT CODES
    # (i, j) = np.where(x > feature_width//2 - 1)
    # x, y = x[i], y[i]
    # (i, j) = np.where(y > feature_width//2 - 1)
    # x, y = x[i], y[i]

    # (i, j) = np.where(x < np.shape(image)[1] - feature_width//2)
    # x, y = x[i], y[i]
    # (i, j) = np.where(y < np.shape(image)[0] - feature_width//2)
    # x, y = x[i], y[i]


    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################  
    fv = np.zeros((np.shape(y)[0], feature_width//4 * feature_width//4 * 8), float)

    image_gradx = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3)
    image_grady = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3)
    image_grad_mag = np.sqrt(image_gradx ** 2 + image_grady ** 2)
    image_grad_mag_blurred = cv2.GaussianBlur(image_grad_mag, (7 , 7), 0) # Best is 7
    image_grad_dir = np.arctan2(-image_grady , image_gradx) * (180/np.pi)
    bin = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

    for numInterestPoints in range(0, len(y)):

        window_mag = image_grad_mag_blurred[int(y[numInterestPoints]) - feature_width//2 : int(y[numInterestPoints]) + feature_width//2 , int(x[numInterestPoints]) - feature_width//2 : int(x[numInterestPoints]) + feature_width//2]
        window_dir = image_grad_dir[int(y[numInterestPoints]) - feature_width//2 : int(y[numInterestPoints]) + feature_width//2 , int(x[numInterestPoints]) - feature_width//2 : int(x[numInterestPoints]) + feature_width//2]
        feature = np.zeros((1, feature_width//4 * feature_width//4 * 8), float)

        for m in range(0, feature_width//4):
            for n in range(0, feature_width//4):
                feature[0, (m * 4 + n) * 8 : (m * 4 + n + 1) * 8] = np.histogram(window_dir[m * 4 : (m + 1) * 4 , n * 4 : (n + 1) * 4], bins = bin, weights = window_mag[m * 4 : (m + 1) * 4 , n * 4 : (n + 1) * 4])[0]

        feature = feature / np.linalg.norm(feature)
        feature = np.clip(feature, a_min = 0, a_max = 0.2)
        feature = feature / np.linalg.norm(feature)
        fv[numInterestPoints, ] = feature

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
