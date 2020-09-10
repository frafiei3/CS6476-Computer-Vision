import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################

    distance = np.zeros((np.shape(features1)[0], np.shape(features2)[0]), float)
    sorted_distance_idx = np.zeros((np.shape(features1)[0], np.shape(features2)[0]), int)
    sorted_distance = np.zeros((np.shape(features1)[0], np.shape(features2)[0]), float)

    for i in range(0, np.shape(features1)[0]):
        dist = np.linalg.norm(np.array(features2) - np.array(features1[i]), axis = 1)
        distance[i,] = dist
        sorted_distance_idx[i,] = np.argsort(dist)
        sorted_distance[i,] = dist[sorted_distance_idx[i,]]

    threshold = 1
    NNDistRatio = sorted_distance[:,0] / sorted_distance[:,1]
    thresh_NNDistRatio_idx = np.array(np.nonzero(NNDistRatio < threshold))

    matches = np.zeros((np.shape(thresh_NNDistRatio_idx)[1], 2), int)
    matches[:, 0] = thresh_NNDistRatio_idx
    matches[:, 1] = sorted_distance_idx[thresh_NNDistRatio_idx, 0]
    confidence = 1 - NNDistRatio[thresh_NNDistRatio_idx]
    sorted_confidence_idx = np.flip(np.argsort(confidence))

    matches[:, 0] = matches[sorted_confidence_idx, 0]
    matches[:, 1] = matches[sorted_confidence_idx, 1]

    confidence = confidence.T
    confidences = confidence[sorted_confidence_idx]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences
