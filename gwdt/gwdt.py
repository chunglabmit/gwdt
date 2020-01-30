import numpy as np
from .gwdt_impl import gwdt_impl


def gwdt(img:np.ndarray, structure:np.ndarray=None) -> np.ndarray:
    """
    Gray-weighted distance transform

    This algorithm finds the weighted manhattan distance from the background
    to every foreground point. The distance is the smallest sum of image values
    along a path. Path steps are taken in directions indicated by the structure.

    :param img: An image where all non-positive values are background and all
    positive values are foreground. The type should be float32.
    :param structure: a structuring element indicating possible path choices
    from the center of the array to its edges. True values indicate possible
    choices where False indicate disallowed choices. This should be of the
    same dimensionality as the image, e.g
    np.ndarray([[False, True, False], [True, True, True], [False, True, False]])
    for a four-connected array in 2 dimensions.
    :param returns an array indicating the distances.
    """
    pad_size = [(_//2, _//2) for _ in structure.shape]
    padded_img = np.pad(img, pad_size).astype(np.float32)
    d = np.mgrid[tuple([slice(-ps[0], ps[1]+1) for ps in pad_size])]
    d = d[:, structure]
    stride = []
    for idx in range(d.shape[1]):
        accumulator = 0
        for idx2 in range(d.shape[0]):
            accumulator += padded_img.strides[idx2] * d[idx2, idx] /\
                           padded_img.dtype.itemsize
        if accumulator != 0:
            stride.append(accumulator)
    strides = np.array(stride, np.int64)
    marks = np.zeros(padded_img.shape, np.uint8)
    # MARK_ALIVE = 1
    # MARK_FAR = 3
    # so False * 2 + 1 = MARK_ALIVE and
    #    True * 2 + 1 = MARK_FAR
    mark_slices = [slice(ps[0], s-ps[1])
                   for ps, s in zip(pad_size, padded_img.shape)]
    marks[tuple(mark_slices)] = (img > 0) * 2 + 1
    output = np.zeros(padded_img.shape, np.float32)
    gwdt_impl(padded_img.ravel(), output.ravel(), strides, marks.ravel())
    return output[tuple(mark_slices)]
