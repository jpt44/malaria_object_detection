import numpy as np


def np_iou(boxA, arrB):
    """
    Calculates the IoUs with respect to all boxes in arrB
    boxA: numpy array of size [1, 4] in order (xmin, ymin, xmax, ymax) where x is along horizontal axis.
          IoUs are calculated with respect to this box

    arrB: numpy array of size [N, 4] where N is the number of bounding boxes detected in an image.
         Entries must be in order: (xmin, ymin, xmax, ymax)

    :returns flat numpy array of IoUs of shape[N,]

    """

    boxA = boxA.reshape(1, 4)
    arrA = np.zeros(shape=arrB.shape, dtype="float32") + boxA

    xAs = np.maximum(arrA[:, 0], arrB[:, 0])  # get the max of the xmin's
    yAs = np.maximum(arrA[:, 1], arrB[:, 1])  # get the max of the ymin's
    xBs = np.minimum(arrA[:, 2], arrB[:, 2])  # get the min of the xmax's
    yBs = np.minimum(arrA[:, 3], arrB[:, 3])  # get the min of the ymax's

    interAreas = np.abs(np.maximum(xBs - xAs, 0) * np.maximum(yBs - yAs, 0))

    areaA = abs((boxA[0, 2] - boxA[0, 0]) * (boxA[0, 3] - boxA[0, 1]))  # width * len  scalar value
    areasB = np.abs((arrB[:, 2] - arrB[:, 0]) * (arrB[:, 3] - arrB[:, 1]))

    return interAreas / (areaA + areasB - interAreas)


def overlap_based_on_iou(boxes, scores, iou_thres=0.5):
    """
    Some objects can have multiple boxes and multiple classes.
    We only want the most confident class and the associated bounding box

    boxes: numpy array containing bounding box coordinates in order (xmin, ymin, xmax, ymax) shape = [N, 4]
    scores: numpy array of shape [N,] which contains the confidence score of class

    rowAssoc: dictionary of rows in bounding box, class, score arrays that need to be deleted
    Rows that need to be deleted are the values in the dictionary, keys are the rows that will be kept
    """

    rowsToConsider = set(range(boxes.shape[0]))
    rowAssoc = dict()

    while len(rowsToConsider) > 0:
        i = rowsToConsider.pop()
        ious = np_iou(boxes[i, :], boxes)  # contains the ious with respect to a given bnd box

        if np.any(ious >= iou_thres, axis=None):

            # find all iou indices that are >=threshold. These boxes overlap
            indices = np.argwhere(ious >= iou_thres).flatten()

            rel_scores = scores[indices]  # relevant confidence scores

            # get index associated with highest confidence score
            final_idx = indices[np.argmax(rel_scores, axis=0),]

            tempSet = set(indices)
            tempSet.discard(final_idx)

            if len(tempSet) > 0:
                rowAssoc[final_idx] = tempSet
                rowsToConsider.difference_update(tempSet)

    return rowAssoc