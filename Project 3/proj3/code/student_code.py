import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    # M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
    #                 [0.6750, 0.3152, 0.1136, 0.0480],
    #                 [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################

    A = []
    B = []

    for i in range(0, np.shape(points_2d)[0]):
        ui, vi = points_2d[i, :]
        Xi, Yi, Zi = points_3d[i, :]
        odd_row = [Xi, Yi, Zi, 1, 0, 0, 0, 0, -ui * Xi, -ui * Yi, -ui * Zi]
        even_row = [0, 0, 0, 0, Xi, Yi, Zi, 1, -vi * Xi, -vi * Yi, -vi * Zi]
        A.append(odd_row)
        A.append(even_row)
        B.append(ui)
        B.append(vi)

    A = np.array(A)
    B = np.array(B)

    M = np.linalg.lstsq(A, B, rcond = -1)[0]
    M = np.append(M, 1)
    M = np.reshape(M, (3, -1))

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    # cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################

    Q = M[:, 0:3]
    m4 = M[:, 3]
    cc = np.matmul(np.linalg.inv(-Q), m4)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

def estimate_fundamental_matrix(points_a, points_b, Normalization = None):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    # F = np.asarray([[0, 0, -0.0004],
    #                 [0, 0, 0.0032],
    #                 [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################

    # Set the normalization as default action
    if Normalization is None:
    	Normalization = True

    # Normalization Function
    def normFunc(POINTS):

        cu, cv = np.sum(POINTS[:, 0])/np.shape(POINTS)[0], np.sum(POINTS[:, 1])/np.shape(POINTS)[0]
        su, sv = 1 / np.std(POINTS[:, 0] - cu), 1 / np.std(POINTS[:, 1] - cv)
        T = np.matmul([[su, 0, 0], [0, sv, 0], [0, 0, 1]], [[1, 0, -cu], [0, 1, -cv], [0, 0, 1]])
        POINTS = np.matmul(T, np.hstack((POINTS, np.ones((np.shape(POINTS)[0], 1)))).T)
        POINTS = POINTS.T

        return POINTS, T


    if Normalization:
    	points_a, T_a = normFunc(points_a)
    	points_b, T_b = normFunc(points_b)
    else:
    	T_a, T_b = np.identity(3), np.identity(3)


    A = []

    for i in range(0, np.shape(points_a)[0]):
        u1 = points_a[i, 0]
        v1 = points_a[i, 1]
        u2 = points_b[i, 0]
        v2 = points_b[i, 1]
        A.append([u1 * u2, v1 * u2, u2, u1 * v2, v1 * v2, v2, u1, v1, 1])

    A = np.array(A)
    
    u, s, vh = np.linalg.svd(A)
    F = np.reshape(vh.T[:, -1], (3, -1))

    u, s, vh = np.linalg.svd(F)
    s[2] = 0
    F = np.matmul(np.matmul(u, np.diag(s)), vh)

    F = np.matmul(np.matmul(T_b.T, F), T_a)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return F

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    # best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    # inliers_a = matches_a[:100, :]
    # inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

    numIterations = 5000
    best_numInliers = 0
    iter = 0

    while iter < numIterations:
    	sample_idx = np.random.choice(np.shape(matches_a)[0], size = 9)
    	# In case you do not want normalization use : F = estimate_fundamental_matrix(matches_a[sample_idx, :], matches_b[sample_idx, :], False) in next line
    	F = estimate_fundamental_matrix(matches_a[sample_idx, :], matches_b[sample_idx, :])
    	inliers_a_iter, inliers_b_iter, error_iter = [], [], []
    	numInliers = 0

    	for i in range(0, np.shape(matches_a)[0]):
    		error = np.matmul(np.matmul(np.append(matches_b[i, :], 1), F), np.append(matches_a[i, :], 1).T)

    		if np.absolute(error) < 0.01:
    			inliers_a_iter.append(matches_a[i, :])
    			inliers_b_iter.append(matches_b[i, :])
    			error_iter.append(error)
    			numInliers = numInliers + 1

    	if numInliers > best_numInliers:
    		best_numInliers = numInliers
    		print(numInliers)
    		best_F = F
    		inliers_a = np.array(inliers_a_iter)
    		inliers_b = np.array(inliers_b_iter)
    		residuals = np.array(error_iter)

    	iter = iter + 1

    inliers_idx = np.argsort(residuals)[:100]
    inliers_a = inliers_a[inliers_idx, :]
    inliers_b = inliers_b[inliers_idx, :]

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_F, inliers_a, inliers_b
