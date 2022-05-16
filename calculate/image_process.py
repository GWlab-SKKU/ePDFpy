import numpy as np
import cv2
import time
import util
import definitions
from scipy import ndimage
from calculate import polar_transform, elliptical_correction

try:
    import cupy as cp
    use_cupy = True
except ImportError:
    use_cupy = False

# mask = np.loadtxt(definitions.MASK_PATH,delimiter=',').astype(np.uint8)
colorcube = (np.loadtxt(definitions.COLORCUBE, delimiter=",", dtype=np.float32) * 255).astype('int')

def draw_center_line(img, center):
    c_x, c_y = center
    c_x = int(c_x)
    c_y = int(c_y)
    rs = img.copy()
    cv2.line(rs, (c_x, 0), (c_x, rs.shape[0]), 255, 5)
    cv2.line(rs, (0, c_y), (rs.shape[1], c_y), 255, 5)
    return rs

###################### calculate center #####################
def calculate_center(img, mask=None):
    center, cost = calculate_center_with_cost(img, mask)
    return center

def calculate_center_with_cost(img, mask):
    image = img.copy()

    # blur
    # image = cv2.GaussianBlur(image, (0,0), 1)

    # initial center
    initial_center = np.round(_calculate_initial_center(image)).astype(int)
    print("initial center is ", initial_center)

    # minimum distance
    search_length = 15
    edge = [[0,image.shape[1]],[image.shape[0],0]]
    minimum_d = np.floor(np.min(np.abs(edge - np.array(initial_center)))).astype(int)
    minimum_d = minimum_d - search_length
    print("minimum_d is",minimum_d)

    # evaluate center
    cost_array = _evaluate_center_local_area(image, initial_center, search_length, minimum_d, mask)

    # recover center index
    min_index = np.unravel_index(cost_array.argmin(), cost_array.shape)
    real_index = np.array(min_index) - search_length
    center = np.round(np.add(initial_center, real_index)).astype('int')
    print("calculated center is ", center)

    return center, cost_array

def _evaluate_center_local_area(img, initial_center, search_length, maximum_d, mask=None):
    """
    :param img: numpy array
    :param initial_center: coordinate tuple
    :param search_length:
    :param maximum_d: when evaluate_center
    :return: search_length*2 x search_length*2
    """
    cost_img = np.zeros((search_length * 2, search_length * 2))
    for x in range(-search_length, search_length):
        for y in range(-search_length, search_length):
            center_xy = (initial_center[0] + x, initial_center[1] + y)
            cost_img[x + search_length, y + search_length] \
                = _evaluate_center(img, center_xy, maximum_d, mask)
    return cost_img

def _evaluate_center(img, center, max_d=None, mask=None):
    dr = 1
    dphi = np.radians(2)

    if mask is not None:
        polar_img = elliptical_correction.polar_transformation_py4d(img, [center[1], center[0]], 1, 1, 0, dr=dr,
                                                                           dphi=dphi, mask=~mask)
    else:
        polar_img = elliptical_correction.polar_transformation_py4d(img, [center[1], center[0]], 1, 1, 0, dr=dr,
                                                                           dphi=dphi)
    norm_std_graph = np.std(polar_img, axis=0)/np.average(polar_img, axis=0)
    if max_d is not None:
        return np.sum(norm_std_graph[:max_d])
    else:
        return np.sum(norm_std_graph)


def calculate_center_gradient(img, mask=None):
    cost_img = np.empty(img.shape)
    cost_img[:] = np.NaN

    # minimum distance
    cursor = np.around(_calculate_initial_center(img)).astype(int)
    print("initial center is ", cursor)

    search_length = 10
    edge = [[0, img.shape[1]], [img.shape[0], 0]]
    minimum_d = np.floor(np.min(np.abs(edge - np.array(cursor)))).astype(int)
    minimum_d = minimum_d - search_length
    print("minimum_d is", minimum_d)

    cnt = 0
    # while (cnt < 15):
    while (cnt < 20):
        search_rect_width = 3
        for x in range(cursor[0] - search_rect_width // 2, cursor[0] + search_rect_width // 2 + 1):
            for y in range(cursor[1] - search_rect_width // 2, cursor[1] + search_rect_width // 2 + 1):
                if not np.isnan(cost_img[x, y]):
                    continue
                cost_img[x, y] = _evaluate_center(img, (x, y), minimum_d, mask)
        #         print("xy loop:",x,y,cost_img[x, y])
        # print(cnt)
        if cost_img[cursor[0],cursor[1]] != np.nanmin(cost_img):
            cursor = np.unravel_index(np.nanargmin(cost_img), cost_img.shape)
            cnt = cnt + 1
        else:
            print("calculated center is",cursor)
            return cursor
    print(f"Failed to find center in {cnt}px")
    return calculate_center(img, mask)


def _calculate_initial_center(img):
    if not len(img.shape) == 2:
        raise ValueError()
    img = cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)
    if img[0][0].dtype != np.uint8:
        img = img.astype(np.uint8)

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]  # it need uint8 as input
    kernel = np.ones((20, 20), np.uint8)
    # thresh = cv2.erode(thresh,kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((200, 200), np.uint8)
    # thresh = cv2.erode(thresh,kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = np.uint8(thresh / 255)
    center_x, center_y = get_CoM(thresh)
    return center_x, center_y


def get_CoM(img):
    grid_x, grid_y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    center_x = np.sum(img * grid_x) / np.sum(img)
    center_y = np.sum(img * grid_y) / np.sum(img)
    return center_x, center_y


def _get_rectangle_from_intensity(image, intensity_range):
    i1, i2 = intensity_range
    kernel_size = 2
    msk = cv2.inRange(image, i1, i2)
    kernel1 = np.ones((kernel_size, kernel_size), np.uint8)
    msk = cv2.erode(msk, kernel1)
    nonzero = np.nonzero(msk)
    p1_y = nonzero[0].min() - 2
    p1_x = nonzero[1].min() - 2
    p2_y = nonzero[0].max() + 2
    p2_x = nonzero[1].max() + 2
    return (p1_y, p1_x, p2_y, p2_x)


def _evaluate_center_slice_range(image, center, rect, value_range, step_size):
    center_x, center_y = center
    center_x = int(center_x)
    center_y = int(center_y)
    min_i, max_i = value_range
    shift_range = 5

    p1_y, p1_x, p2_y, p2_x = rect

    y_width_origin = p2_y - p1_y
    x_width_origin = p2_x - p1_x
    h_y_width_origin = int(y_width_origin / 2)
    h_x_width_origin = int(x_width_origin / 2)

    p1y = np.max((center_y - h_y_width_origin, 0)).astype(np.uint16)
    p2y = np.min((center_y + h_y_width_origin, image.shape[0])).astype(np.uint16)
    p1x = np.max((center_x - h_x_width_origin, 0)).astype(np.uint16)
    p2x = np.min((center_x + h_x_width_origin, image.shape[1])).astype(np.uint16)

    y_width = p2_y - p1_y
    x_width = p2_x - p1_x
    h_y_width = int(y_width / 2)
    h_x_width = int(x_width / 2)

    slc_img = image[p1y:p2y, p1x:p2x]
    slc_beam_mask = mask[p1y:p2y, p1x:p2x]

    # slc_beam_mask = beam_mask[center_y-h_y_width:center_y+h_y_width,center_x-h_x_width:center_x+h_x_width]
    slc_center_y = h_y_width
    slc_center_x = h_x_width
    slc_center_y = center_y - p1y
    slc_center_x = center_x - p1x

    if step_size > 0:
        range_ = np.linspace(min_i, max_i, step_size + 1)
    std_sum = 0
    for step in range(len(range_) - 1):
        current_min_i = int(range_[step])
        current_max_i = int(range_[step + 1])

        ring_mask = cv2.inRange(slc_img, current_min_i, current_max_i)
        ring_mask = cv2.bitwise_and(ring_mask, ring_mask, mask=(np.logical_not(slc_beam_mask) * 255).astype(np.uint8))

        mesh = np.meshgrid(range(slc_img.shape[1]), range(slc_img.shape[0]))
        mesh_y = mesh[1] - slc_center_y
        mesh_x = mesh[0] - slc_center_x

        ring_x = cv2.bitwise_and(mesh_x, mesh_x, mask=ring_mask)
        ring_y = cv2.bitwise_and(mesh_y, mesh_y, mask=ring_mask)

        dt = np.power(np.square(ring_x) + np.square(ring_y), 0.5)
        dt = dt.reshape(-1)
        std_sum += np.std(dt[np.nonzero(dt)[0]])

    return std_sum


def _calculate_azimuthal_average_deprecated(raw_image, center):
    # if use_cupy:
    #     return calculate_azimuthal_average_cuda(raw_image, center)
    center_x, center_y = center
    mesh = np.meshgrid(range(raw_image.shape[1]), range(raw_image.shape[0]))
    mesh_x = mesh[0] - center_x
    mesh_y = mesh[1] - center_y
    rr = np.power(np.square(mesh_x) + np.square(mesh_y), 0.5)
    rr = cv2.bitwise_and(rr, rr, mask=np.bitwise_not(mask))
    n_rr = np.uint16(np.ceil(rr.max()))

    def vector_oper(r):
        masked_img = raw_image[(rr >= r - .5) & (rr < r + .5)]
        return masked_img.mean(), masked_img.var()

    # f = lambda r: raw_image[(rr >= r - .5) & (rr < r + .5)].mean()
    r = np.linspace(1, n_rr, num=n_rr)
    mean, var = np.vectorize(vector_oper)(r)
    mean = np.nan_to_num(mean, 0)
    var = np.nan_to_num(var, 0)

    return mean, var


def calculate_azimuthal_average_with_std(raw_image, center):
    raw_min = raw_image.min()
    raw_image_abs = raw_image.copy()
    if raw_min < 0:
        raw_image_abs = raw_image_abs+np.abs(raw_min)+1

    mesh = np.meshgrid(range(raw_image.shape[1]), range(raw_image.shape[0]))
    mesh_x = mesh[0] - center[0]
    mesh_y = mesh[1] - center[1]
    rr = np.power(np.square(mesh_x) + np.square(mesh_y), 0.5)
    rr = cv2.bitwise_and(rr, rr, mask=np.bitwise_not(mask))
    rr = np.rint(rr).astype('uint16')
    n_rr = np.uint16(rr.max())


    #### radial mean ####
    radial_mean = ndimage.mean(raw_image, labels=rr, index=np.arange(0, n_rr + 1))
    radial_mean = np.nan_to_num(radial_mean, 0)

    #### std ####
    # todo: algo duplicated
    corrected_mean = ndimage.mean(raw_image_abs, labels=rr, index=np.arange(0, n_rr + 1))
    std = ndimage.standard_deviation(raw_image_abs, labels=rr, index=np.arange(0, n_rr + 1))

    first_peak = 0
    for i in range(len(radial_mean)):
        if radial_mean[i] != 0:
            first_peak = i
            break

    normalized_std = np.zeros(radial_mean.shape)
    normalized_std[first_peak:] = std[first_peak:] / corrected_mean[first_peak:]
    return radial_mean, normalized_std


def calculate_azimuthal_average(raw_image, center, mask=None):
    raw_min = raw_image.min()
    raw_image_abs = raw_image.copy()
    if raw_min < 0:
        raw_image_abs = raw_image_abs+np.abs(raw_min)+1

    mesh = np.meshgrid(range(raw_image.shape[1]), range(raw_image.shape[0]))
    mesh_x = mesh[0] - center[0]
    mesh_y = mesh[1] - center[1]
    rr = np.power(np.square(mesh_x) + np.square(mesh_y), 0.5)
    if mask is not None:
        rr = cv2.bitwise_and(rr, rr, mask=np.bitwise_not(mask))
    else:
        rr = cv2.bitwise_and(rr, rr)
    rr = np.rint(rr).astype('uint16')
    n_rr = np.uint16(rr.max())

    #### radial mean ####
    # radial_mean = ndimage.mean(raw_image, labels=rr, index=np.arange(0, n_rr + 1))
    radial_mean = ndimage.mean(raw_image, labels=rr, index=np.arange(1, n_rr + 1))
    radial_mean = np.insert(radial_mean, 0, 0)
    radial_mean = np.nan_to_num(radial_mean, 0)

    return radial_mean


def calculate_azimuthal_average_ellipse(raw_image, center, ellipse_p, mask=None):
    rs = elliptical_correction._cartesian_to_polarelliptical_transform(raw_image, [*center,*ellipse_p], mask)
    rr = rs[1]
    if mask is not None:
        rr = cv2.bitwise_and(rr, rr, mask=np.bitwise_not(mask))
    else:
        rr = cv2.bitwise_and(rr, rr)
    rr = np.rint(rr).astype('uint16')
    n_rr = np.uint16(rr.max())

    #### radial mean ####
    # radial_mean = ndimage.mean(raw_image, labels=rr, index=np.arange(0, n_rr + 1))
    radial_mean = ndimage.mean(raw_image, labels=rr, index=np.arange(1, n_rr + 1))
    radial_mean = np.insert(radial_mean, 0, 0)
    radial_mean = np.nan_to_num(radial_mean, 0)

    return radial_mean

def calculate_azimuthal_average_(raw_image, center):
    raw_image = raw_image.copy()
    mesh = np.meshgrid(range(raw_image.shape[1]), range(raw_image.shape[0]))
    mesh_x = mesh[0] - center[0]
    mesh_y = mesh[1] - center[1]
    rr = np.power(np.square(mesh_x) + np.square(mesh_y), 0.5)
    rr = cv2.bitwise_and(rr, rr, mask=np.bitwise_not(mask))
    rr = np.rint(rr).astype('uint16')
    n_rr = np.uint16(rr.max())

    count, sum, sum_c_sq = _stats(raw_image, labels=rr, index=np.arange(0, n_rr + 1), centered=True)

    #### radial mean ####
    radial_mean = sum / np.asanyarray(count).astype(np.float64)
    radial_mean = np.nan_to_num(radial_mean, 0)

    radial_mean_corrected = np.zeros(radial_mean.shape)
    radial_min = np.min(radial_mean)
    if radial_min <= 0:
        radial_mean_corrected = radial_mean+np.abs(radial_min)+1

    #### std ####
    std = sum_c_sq / np.asanyarray(count).astype(float)

    first_peak = 0
    for i in range(len(radial_mean)):
        if radial_mean[i] != 0:
            first_peak = i
            break
    normalized_std = np.zeros(radial_mean.shape)
    normalized_std[first_peak:] = std[first_peak:] / radial_mean_corrected[first_peak:]
    return radial_mean, normalized_std

def _stats(input, labels=None, index=None, centered=False):
    """Count, sum, and optionally compute (sum - centre)^2 of input by label

    Parameters
    ----------
    input : array_like, N-D
        The input data to be analyzed.
    labels : array_like (N-D), optional
        The labels of the data in `input`. This array must be broadcast
        compatible with `input`; typically, it is the same shape as `input`.
        If `labels` is None, all nonzero values in `input` are treated as
        the single labeled group.
    index : label or sequence of labels, optional
        These are the labels of the groups for which the stats are computed.
        If `index` is None, the stats are computed for the single group where
        `labels` is greater than 0.
    centered : bool, optional
        If True, the centered sum of squares for each labeled group is
        also returned. Default is False.

    Returns
    -------
    counts : int or ndarray of ints
        The number of elements in each labeled group.
    sums : scalar or ndarray of scalars
        The sums of the values in each labeled group.
    sums_c : scalar or ndarray of scalars, optional
        The sums of mean-centered squares of the values in each labeled group.
        This is only returned if `centered` is True.

    """
    def single_group(vals):
        if centered:
            vals_c = vals - vals.mean()
            return vals.size, vals.sum(), (vals_c * vals_c.conjugate()).sum()
        else:
            return vals.size, vals.sum()

    if labels is None:
        return single_group(input)

    # ensure input and labels match sizes
    input, labels = np.broadcast_arrays(input, labels)

    if index is None:
        return single_group(input[labels > 0])

    if np.isscalar(index):
        return single_group(input[labels == index])

    def _sum_centered(labels):
        # `labels` is expected to be an ndarray with the same shape as `input`.
        # It must contain the label indices (which are not necessarily the labels
        # themselves).
        means = sums / counts
        centered_input = input - means[labels]
        # bincount expects 1-D inputs, so we ravel the arguments.
        bc = np.bincount(labels.ravel(),
                              weights=(centered_input *
                                       centered_input.conjugate()).ravel())
        return bc

    # Remap labels to unique integers if necessary, or if the largest
    # label is larger than the number of values.

    if (not _safely_castable_to_int(labels.dtype) or
            labels.min() < 0 or labels.max() > labels.size):
        # Use np.unique to generate the label indices.  `new_labels` will
        # be 1-D, but it should be interpreted as the flattened N-D array of
        # label indices.
        unique_labels, new_labels = np.unique(labels, return_inverse=True)
        counts = np.bincount(new_labels)
        sums = np.bincount(new_labels, weights=input.ravel())
        if centered:
            # Compute the sum of the mean-centered squares.
            # We must reshape new_labels to the N-D shape of `input` before
            # passing it _sum_centered.
            sums_c = _sum_centered(new_labels.reshape(labels.shape))
        idxs = np.searchsorted(unique_labels, index)
        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = (unique_labels[idxs] == index)
    else:
        # labels are an integer type allowed by bincount, and there aren't too
        # many, so call bincount directly.
        counts = np.bincount(labels.ravel())
        sums = np.bincount(labels.ravel(), weights=input.ravel())
        if centered:
            sums_c = _sum_centered(labels)
        # make sure all index values are valid
        idxs = np.asanyarray(index, np.int_).copy()
        found = (idxs >= 0) & (idxs < counts.size)
        idxs[~found] = 0

    counts = counts[idxs]
    counts[~found] = 0
    sums = sums[idxs]
    sums[~found] = 0

    if not centered:
        return (counts, sums)
    else:
        sums_c = sums_c[idxs]
        sums_c[~found] = 0
        return (counts, sums, sums_c)

def _safely_castable_to_int(dt):
    """Test whether the NumPy data type `dt` can be safely cast to an int."""
    int_size = np.dtype(int).itemsize
    safe = ((np.issubdtype(dt, np.signedinteger) and dt.itemsize <= int_size) or
            (np.issubdtype(dt, np.unsignedinteger) and dt.itemsize < int_size))
    return safe


def _calculate_azimuthal_average_cuda_deprecated(raw_image, center):
    img = cp.array(raw_image)
    beam = cp.array(mask)
    center_x, center_y = center

    mesh = cp.meshgrid(cp.arange(raw_image.shape[1]), cp.arange(raw_image.shape[0]))
    mesh_x = mesh[0] - center_x
    mesh_y = mesh[1] - center_y
    rr = cp.power(cp.square(mesh_x) + cp.square(mesh_y), 0.5)
    cp.putmask(rr, beam, 0)
    n_rr = int(cp.ceil(rr.max()))

    azav = cp.zeros((n_rr + 1))
    azvar = cp.zeros((n_rr + 1))
    for n in range(1, n_rr):
        rig_mask = (rr >= n - 0.5) & (rr < n + 0.5)
        rig = img[rig_mask]
        azav[n] = rig.mean()
        azvar[n] = rig.var()

        # rig_mask = (rr >= n - 0.5) & (rr < n + 0.5)
        # rig = img * rig_mask
        # rig = rig.reshape((-1))
        # non_zero = cp.nonzero(rig)[0]
        # ring_non_zero = rig[non_zero]
        # azav[n] = cp.average(ring_non_zero)
        # azvar[n] = cp.var(ring_non_zero)

    azav = np.nan_to_num(azav.get(), 0)
    azvar = np.nan_to_num(azvar.get(), 0)

    return azav, azvar
