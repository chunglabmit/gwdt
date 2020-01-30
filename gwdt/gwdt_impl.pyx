# distutils: language = c++
# cython: boundscheck = False
#
# The GWDT algorithm is taken from
# Xiao H, Peng H. APP2: automatic tracing of 3D neuron morphology based on
# hierarchical pruning of a gray-weighted image distance-tree.
# Bioinformatics. 2013;29(11):1448–1454. doi:10.1093/bioinformatics/btt170
#

from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from libcpp.utility cimport pair as cpp_pair
cimport numpy as np
import numpy as np
cimport cython

#
# The priority queue's top is the maximal element, so the weights
# for the first element of the pair should be negative
#
ctypedef cpp_pair[np.float32_t, np.uint64_t] queue_elem_t
ctypedef priority_queue[queue_elem_t] queue_t

cdef:
    np.uint8_t MARK_OOB = 0
    np.uint8_t MARK_ALIVE = 1
    np.uint8_t MARK_TRIAL = 2
    np.uint8_t MARK_FAR = 3

np.import_array()

def gwdt_impl(np.float32_t[:] img,
              np.float32_t[:] output,
              np.int64_t[:] strides,
              np.uint8_t[:] marks):
     """Grey-weighted distance transform implementation

     The grey-weighted distance from the background to each foreground
     voxel. Steps are taken along one of the directions indicated by the
     strides, accumulating the image intensity as the distance. Voxels that
     are less than or equal to zero are background.

     :param img: the intensity image, normalized so that background voxels
     are zero or lower (e.g. subtract the threshold from the image). The image
     should be pre-padded so that the edges are extended (e.g. by np.pad)
     and then flattened.
     :param output: an array of size and shape like the image. This will be
     filled with the distance to every point.
     :param strides: each element of this array is a step size from a voxel
     to a connected voxel. For example, for a 1024 x 1024 image (with padding),
     the strides might be 1, -1, 1024 and -1024 to connect X and Y adjacent
     pixels.
     :param marks: this is a working space for the algorithm of the same
     shape as the image. Initially, out-of-bounds pixels should be marked as
     zero (MARK_OOB - e.g. the edges), background pixels should be marked
     as 1 (MARK_ALIVE) and foreground pixels should be marked as 3 (MARK_FAR).
     The algorithm will use the space to keep track of its progress and
     the array can otherwise be ignored after initialization.
     """
     cdef:
         queue_t queue
     with nogil:
         initialize(img, output, strides, marks, queue)
         while queue.size() > 0:
            do_gwdt_step(img, output, strides, marks, queue)

cdef void initialize(np.float32_t[:] img,
                     np.float32_t[:] output,
                     np.int64_t[:] strides,
                     np.uint8_t[:] marks,
                     queue_t &queue) nogil:
    cdef:
       np.uint64_t idx
       np.uint64_t stride_idx
       np.uint64_t trial_idx
       queue_elem_t trial_elem
    for 0 <= idx < img.shape[0]:
        if marks[idx] == MARK_OOB:
            continue
        elif marks[idx] == MARK_ALIVE:
            for 0 <= stride_idx < strides.shape[0]:
                trial_idx = idx + strides[stride_idx]
                if marks[trial_idx] == MARK_FAR:
                     trial_elem.first = -img[trial_idx]
                     trial_elem.second = trial_idx
                     queue.push(trial_elem)
                     marks[trial_idx] = MARK_TRIAL

cdef void do_gwdt_step(np.float32_t[:] img,
                        np.float32_t[:] output,
                        np.int64_t[:] strides,
                        np.uint8_t[:] marks,
                        queue_t &queue) nogil:
    cdef:
        queue_elem_t trial_elem
        np.uint64_t stride_idx
        np.uint64_t trial_idx
        np.float32_t weight
        np.uint64_t candidate_idx
        np.uint8_t candidate_mark
        queue_elem_t candidate_elem

    trial_elem = queue.top()
    queue.pop()
    weight = -trial_elem.first
    trial_idx = trial_elem.second
    if marks[trial_idx] != MARK_ALIVE or output[trial_idx] > weight:
        # A better path has already been found if MARK_ALIVE
        marks[trial_idx] = MARK_ALIVE
        output[trial_idx] = weight
        for 0 <= stride_idx < strides.shape[0]:
            candidate_idx = trial_idx + strides[stride_idx]
            candidate_mark = marks[candidate_idx]
            if candidate_mark == MARK_TRIAL or candidate_mark == MARK_FAR:
                # For trial or far-marked candidates, calculate the weight
                # of this path and push it - there may be a better one on
                # the queue, but it will pop out first
                #
                candidate_elem.first = -(weight + img[candidate_idx])
                candidate_elem.second = candidate_idx
                queue.push(candidate_elem)
                marks[candidate_idx] = MARK_TRIAL

