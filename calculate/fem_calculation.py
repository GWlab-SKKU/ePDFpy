

"""
Functions related to elliptical calibration, such as fitting elliptical
distortions.
The user-facing representation of ellipses is in terms of the following 5
parameters:
    x0,y0       the center of the ellipse
    a           the semimajor axis length
    b           the semiminor axis length
    theta       the (positive, right handed) tilt of the a-axis
                to the x-axis, in radians
More details about the elliptical parameterization used can be found in
the module docstring for process/utils/elliptical_coords.py.
"""
import time
import numpy as np
from scipy.optimize import leastsq,curve_fit
from itertools import product
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

###### Fitting a 1d elliptical curve to a 2d array, e.g. a Bragg vector map ######

def ellipse_err(p, x, y, val):
    """
    For a point (x,y) in a 2d cartesian space, and a function taking the value
    val at point (x,y), and some 1d ellipse in this space given by
            ``A(x-x0)^2 + B(x-x0)(y-y0) + C(y-y0)^2 = 1``
    this function computes the error associated with the function's value at (x,y)
    given by its deviation from the ellipse times val.
    Note that this function is for internal use, and uses ellipse parameters `p`
    given in canonical form (x0,y0,A,B,C), which is different from the ellipse
    parameterization used in all the user-facing functions, for reasons of
    numerical stability.
    """
    x,y = x-p[0],y-p[1]
    return (p[2]*x**2 + p[3]*x*y + p[4]*y**2 - 1)*val


###### Fitting from amorphous diffraction rings ######

def fit_ellipse_amorphous_ring(data,center,fitradii,p0=None,mask=None):
    """
    Fit the amorphous halo of a diffraction pattern, including any elliptical distortion.
    The fit function is::
        f(x,y; I0,I1,sigma0,sigma1,sigma2,c_bkgd,x0,y0,A,B,C) =
            Norm(r; I0,sigma0,0) +
            Norm(r; I1,sigma1,R)*Theta(r-R)
            Norm(r; I1,sigma2,R)*Theta(R-r) + c_bkgd
    where
        * (x,y) are cartesian coordinates,
        * r is the radial coordinate,
        * (I0,I1,sigma0,sigma1,sigma2,c_bkgd,x0,y0,R,B,C) are parameters,
        * Norm(x;I,s,u) is a gaussian in the variable x with maximum amplitude I,
          standard deviation s, and mean u
        * Theta(x) is a Heavyside step function
        * R is the radial center of the double sided gaussian, derived from (A,B,C)
          and set to the mean of the semiaxis lengths
    The function thus contains a pair of gaussian-shaped peaks along the radial
    direction of a polar-elliptical parametrization of a 2D plane. The first gaussian is
    centered at the origin. The second gaussian is centered about some finite R, and is
    'two-faced': it's comprised of two half-gaussians of different standard deviations,
    stitched together at their mean value of R. This Janus (two-faced ;p) gaussian thus
    comprises an elliptical ring with different inner and outer widths.
    The parameters of the fit function are
        * I0: the intensity of the first gaussian function
        * I1: the intensity of the Janus gaussian
        * sigma0: std of first gaussian
        * sigma1: inner std of Janus gaussian
        * sigma2: outer std of Janus gaussian
        * c_bkgd: a constant offset
        * x0,y0: the origin
        * A,B,C: The ellipse parameters, in the form Ax^2 + Bxy + Cy^2 = 1
    Args:
        data (2d array): the data
        center (2-tuple of numbers): the center (x0,y0)
        fitradii (2-tuple of numbers): the inner and outer radii of the fitting annulus
        p0 (11-tuple): initial guess parameters. If p0 is None, the function will compute
            a guess at all parameters. If p0 is a 11-tuple it must be populated by some
            mix of numbers and None; any parameters which are set to None will be guessed
            by the function.  The parameters are the 11 parameters of the fit function
            described above, p0 = (I0,I1,sigma0,sigma1,sigma2,c_bkgd,x0,y0,A,B,C).
            Note that x0,y0 are redundant; their guess values are the x0,y0 values passed
            to the main function, but if they are passed as elements of p0 these will
            take precendence.
        mask (2d array of bools): only fit to datapoints where mask is True
    Returns:
        (2-tuple comprised of a 5-tuple and an 11-tuple): Returns a 2-tuple.
        The first element is the ellipse parameters need to elliptically parametrize
        diffraction space, and is itself a 5-tuple:
            * **x0**: x center
            * **y0**: y center,
            * **a**: the semimajor axis length
            * **b**: the semiminor axis length
            * **theta**: tilt of a-axis w.r.t x-axis, in radians
        The second element is the full set of fit parameters to the double sided gaussian
        function, described above, and is an 11-tuple
    """
    data = np.array(data)
    if mask is None:
        mask = np.ones_like(data)
    assert data.shape == mask.shape, "data and mask must have same shapes."
    x0,y0 = center
    ri,ro = fitradii

    # Get data mask
    Nx,Ny = data.shape
    yy,xx = np.meshgrid(np.arange(Ny),np.arange(Nx))
    rr = np.hypot(xx-x0,yy-y0)
    _mask = ((rr>ri)*(rr<ro)).astype(bool)
    _mask *= mask.astype(bool)

    # Make coordinates, get data values
    x_inds, y_inds = np.nonzero(_mask)
    vals = data[_mask]

    # Get initial parameter guesses
    I0 = np.max(data)
    I1 = np.max(data*mask)
    sigma0 = ri/2.
    sigma1 = (ro-ri)/4.
    sigma2 = (ro-ri)/4.
    c_bkgd = np.min(data)
    # To guess R, we take a radial integral
    q,radial_profile = radial_integral(data,x0,y0,1)
    R = q[(q>ri)*(q<ro)][np.argmax(radial_profile[(q>ri)*(q<ro)])]
    # Initial guess at A,B,C
    A,B,C = convert_ellipse_params_r(R,R,0)

    # Populate initial parameters
    p0_guess = tuple([I0,I1,sigma0,sigma1,sigma2,c_bkgd,x0,y0,A,B,C])
    if p0 is None:
        _p0 = p0_guess
    else:
        assert len(p0) == 11
        _p0 = tuple([p0_guess[i] if p0[i] is None else p0[i] for i in range(len(p0))])

    # Perform fit
    p = leastsq(double_sided_gaussian_fiterr, _p0, args=(x_inds, y_inds, vals))[0]

    # Return
    _x0,_y0 = p[6],p[7]
    _A,_B,_C = p[8],p[9],p[10]
    # _a,_b,_theta = convert_ellipse_params(_A,_B,_C)
    # return (_x0, _y0, _a, _b, _theta), p
    return (_x0,_y0,_A,_B,_C),p

def double_sided_gaussian_fiterr(p, x, y, val):
    """
    Returns the fit error associated with a point (x,y) with value val, given parameters p.
    """
    return double_sided_gaussian(p, x, y) - val


def double_sided_gaussian(p, x, y):
    """
    Return the value of the double-sided gaussian function at point (x,y) given
    parameters p, described in detail in the fit_ellipse_amorphous_ring docstring.
    """
    # Unpack parameters
    I0, I1, sigma0, sigma1, sigma2, c_bkgd, x0, y0, A, B, C = p
    a,b,theta = convert_ellipse_params(A,B,C)
    R = np.mean((a,b))
    R2 = R**2
    A,B,C = A*R2,B*R2,C*R2
    r2 = A*(x - x0)**2 + B*(x - x0)*(y - y0) + C*(y - y0)**2
    r = np.sqrt(r2) - R

    return (
        I0 * np.exp(-r2 / (2 * sigma0 ** 2))
        + I1 * np.exp(-r ** 2 / (2 * sigma1 ** 2)) * np.heaviside(-r, 0.5)
        + I1 * np.exp(-r ** 2 / (2 * sigma2 ** 2)) * np.heaviside(r, 0.5)
        + c_bkgd
    )

def radial_elliptical_integral(ar, dr, p_ellipse):
    """
    Computes the radial integral of array ar from center (x0,y0) with a step size in r of
    dr.
    Args:
        ar (2d array): the data
        dr (number): the r sampling
        p_ellipse (5-tuple): the parameters (x0,y0,a,b,theta) for the ellipse
    Returns:
        (2-tuple): A 2-tuple containing:
            * **rbin_centers**: *(1d array)* the bins centers of the radial integral
            * **radial_integral**: *(1d array)* the radial integral
        radial_integral (1d array) the radial integral
    """
    x0, y0 = p_ellipse[0], p_ellipse[1]
    rmax = int(
        max(
            (
                np.hypot(x0, y0),
                np.hypot(x0, ar.shape[1] - y0),
                np.hypot(ar.shape[0] - x0, y0),
                np.hypot(ar.shape[0] - x0, ar.shape[1] - y0),
            )
        )
    )
    polarAr, rr, pp = cartesian_to_polarelliptical_transform(
        ar, p_ellipse=p_ellipse, dr=dr, dphi=np.radians(2), r_range=rmax
    )
    radial_integral = np.sum(polarAr, axis=0)
    rbin_centers = rr[0, :]
    return rbin_centers,radial_integral


def radial_integral(ar, x0, y0, dr):
    """
    Computes the radial integral of array ar from center (x0,y0) with a step size in r of dr.
    Args:
        ar (2d array): the data
        x0,y0 (floats): the origin
        dr (number): radial step size
    Returns:
        (2-tuple): A 2-tuple containing:
            * **rbin_centers**: *(1d array)* the bins centers of the radial integral
            * **radial_integral**: *(1d array)* the radial integral
    """
    return radial_elliptical_integral(ar, dr, (x0,y0,1,1,0))

def convert_ellipse_params_r(a,b,theta):
    """
    Converts from ellipse parameters (a,b,theta) to (A,B,C).
    See module docstring for more info.
    Args:
        a,b,theta (floats): parameters of an ellipse, where `a`/`b` are the
            semimajor/semiminor axis lengths, and theta is the tilt of the semimajor axis
            with respect to the x-axis, in radians.
    Returns:
        (3-tuple): A 3-tuple consisting of (A,B,C), the ellipse parameters in
            canonical form.
    """
    sin2,cos2 = np.sin(theta)**2,np.cos(theta)**2
    a2,b2 = a**2,b**2
    A = sin2/b2 + cos2/a2
    C = cos2/b2 + sin2/a2
    B = 2*(b2-a2)*np.sin(theta)*np.cos(theta)/(a2*b2)
    return A,B,C


def convert_ellipse_params(A,B,C):
    """
    Converts ellipse parameters from canonical form (A,B,C) into semi-axis lengths and
    tilt (a,b,theta).
    See module docstring for more info.
    Args:
        A,B,C (floats): parameters of an ellipse in the form:
                             Ax^2 + Bxy + Cy^2 = 1
    Returns:
        (3-tuple): A 3-tuple consisting of:
        * **a**: (float) the semimajor axis length
        * **b**: (float) the semiminor axis length
        * **theta**: (float) the tilt of the ellipse semimajor axis with respect to
          the x-axis, in radians
    """
    val = np.sqrt((A-C)**2+B**2)
    b4a = B**2 - 4*A*C
    # Get theta
    if B == 0:
        if A<C:
            theta = 0
        else:
            theta = np.pi/2.
    else:
        theta = np.arctan2((C-A-val),B)
    # Get a,b
    a = - np.sqrt( -2*b4a*(A+C+val) ) / b4a
    b = - np.sqrt( -2*b4a*(A+C-val) ) / b4a
    a,b = max(a,b),min(a,b)
    return a,b,theta


def cartesian_to_polarelliptical_transform(
    cartesianData,
    p_ellipse,
    dr=1,
    dphi=np.radians(2),
    r_range=None,
    mask=None,
    maskThresh=0.99,
):
    if mask is None:
        mask = np.ones_like(cartesianData, dtype=bool)
    assert (
        cartesianData.shape == mask.shape
    ), "Mask and cartesian data array shapes must match."
    assert len(p_ellipse) == 5, "p_ellipse must have length 5"

    # Get params
    qx0, qy0, a, b, theta = p_ellipse
    Nx, Ny = cartesianData.shape

    # Define r_range:
    if r_range is None:
        #find corners of image
        corners = np.array([
                            [0,0],
                            [0,cartesianData.shape[0]],
                            [0,cartesianData.shape[1]],
                            [cartesianData.shape[0], cartesianData.shape[1]]
                            ])
        #find maximum corner distance
        r_min, r_max =0, np.ceil(
                            np.max(
                                np.sqrt(
                                    np.sum((corners -np.broadcast_to(np.array((qx0,qy0)), corners.shape))**2, axis = 1)
                                       )
                                   )
                                ).astype(int)
    else:
        try:
            r_min, r_max = r_range[0], r_range[1]
        except TypeError:
            r_min, r_max = 0, r_range

    # Define the r/phi coords
    r_bins = np.arange(r_min + dr / 2.0, r_max + dr / 2.0, dr)  # values are bin centers
    p_bins = np.arange(-np.pi + dphi / 2.0, np.pi + dphi / 2.0, dphi)
    rr, pp = np.meshgrid(r_bins, p_bins)
    Nr, Np = rr.shape

    # Get (qx,qy) corresponding to each (r,phi) in the newly defined coords
    xr = rr * np.cos(pp)
    yr = rr * np.sin(pp)
    qx = qx0 + xr * np.cos(theta) - yr * (b/a) * np.sin(theta)
    qy = qy0 + xr * np.sin(theta) + yr * (b/a) * np.cos(theta)

    # qx,qy are now shape (Nr,Np) arrays, such that (qx[r,phi],qy[r,phi]) is the point
    # in cartesian space corresponding to r,phi.  We now get the values for the final
    # polarEllipticalData array by interpolating values at these coords from the original
    # cartesianData array.

    transform_mask = (qx > 0) * (qy > 0) * (qx < Nx - 1) * (qy < Ny - 1)
    # Bilinear interpolation
    xF = np.floor(qx[transform_mask])
    yF = np.floor(qy[transform_mask])
    dx = qx[transform_mask] - xF
    dy = qy[transform_mask] - yF
    x_inds = np.vstack((xF, xF + 1, xF, xF + 1)).astype(int)
    y_inds = np.vstack((yF, yF, yF + 1, yF + 1)).astype(int)
    weights = np.vstack(
        ((1 - dx) * (1 - dy), (dx) * (1 - dy), (1 - dx) * (dy), (dx) * (dy))
    )
    transform_mask = transform_mask.ravel()
    polarEllipticalData = np.zeros(Nr * Np)
    polarEllipticalData[transform_mask] = np.sum(
        cartesianData[x_inds, y_inds] * weights, axis=0
    )
    polarEllipticalData = np.reshape(polarEllipticalData, (Nr, Np))

    # Transform mask
    polarEllipticalMask = np.zeros(Nr * Np)
    polarEllipticalMask[transform_mask] = np.sum(mask[x_inds, y_inds] * weights, axis=0)
    polarEllipticalMask = np.reshape(polarEllipticalMask, (Nr, Np))

    polarEllipticalData = np.ma.array(
        data=polarEllipticalData, mask=polarEllipticalMask < maskThresh
    )
    return polarEllipticalData, rr, pp


def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    array([9, 7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
            [[0,0],[0,0],[0,1]],
            [[0,0],[0,0],[0,1]],
            [[1,0],[1,0],[1,1]],
        ])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out

def autofit(dc, mask=None):
    initial_center = [dc.shape[2]//2,dc.shape[1]//2]
    if mask is not None:
        idxs = np.where(np.sum(mask, axis=0) > 0)
        percent1 = dc.shape[1] // 100
        inner_rad = idxs[0][0] + percent1
        outer_rad = idxs[0][-1] - percent1
    else:
        # longest point
        inner_rad = 0
        outer_rad = np.min(initial_center)
    fitting_img = np.median(dc, axis=0)
    ellipse_rs = fit_ellipse_amorphous_ring(fitting_img.data, center=initial_center, mask=mask, fitradii=(inner_rad,outer_rad))
    return ellipse_rs[0]

def get_FEM(dc, mask, elliptical_p):
    # FEM42
    ringInt = np.zeros(dc.shape[0])
    for i in range(len(ringInt)):
        ringInt[i] = dc[i][mask.astype(bool)].mean()

    scaleInt = np.mean(ringInt) / ringInt
    for i in range(len(ringInt)):
        dc[i] = dc[i]*scaleInt[i]
    CBEDmean = np.median(dc, 0)  # have to use median?

    # FEM44
    polarCBEDmean, _, _ = cartesian_to_polarelliptical_transform(CBEDmean, elliptical_p, mask=mask)
    polar_dc = np.zeros([dc.shape[0], polarCBEDmean.shape[0], polarCBEDmean.shape[1]])
    polar_dc = np.ma.array(data=polar_dc)
    for i in range(dc.shape[0]):
        polar_dc[i] = cartesian_to_polarelliptical_transform(dc[i], elliptical_p, mask=mask)[0]

    # FEM45
    polarCBEDvar = np.mean((polar_dc - polarCBEDmean) ** 2, axis=0)
    radialVar = np.mean(polarCBEDvar,0)
    radialMean = np.mean(polarCBEDmean,0)
    radialVarNorm = radialVar / radialMean ** 2

    return radialVarNorm


def elliptical_fitting_matlab(image, mask=None):
    def fit_func(x, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
        Rsq = (x[:, 0] - c1) ** 2 + c4 * (x[:, 0] - c1) * (x[:, 1] - c2) + c3 * (x[:, 1] - c2) ** 2
        rs = c5 + \
             c6 * np.exp(-1 / 2 / c7 ** 2 * Rsq) + \
             c9 * np.exp(-1 / 2 / c10 ** 2 * np.abs(c8 - np.sqrt(Rsq)) ** 2) * (c8 ** 2 > Rsq) + \
             c9 * np.exp(-1 / 2 / c11 ** 2 * np.abs(c8 - np.sqrt(Rsq)) ** 2) * (c8 ** 2 < Rsq)
        return rs

    if mask is None:
        mask = np.ones(image.shape)
    skipFit = [11, 1]
    stackSize = image.shape
    ya, xa = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    inds2D = np.ravel_multi_index([ya, xa], stackSize[1:3])
    basis = np.array((ya.reshape(-1), xa.reshape(-1))).transpose()
    p0 = 1.0e+03 * np.array([0.2565, 0.2624, 0.0010, 0.0001, 0.0, 1.4273, 0.0535,
                             0.1198, 0.2895, 0.0118, 0.0104])
    lb = [0, 0, 0.5, -0.5,
          0, 0, 0, 0, 0, 0, 0];
    ub = [stackSize[1], stackSize[2], 2, 0.5
        , np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf];
    coefsInit = p0
    for a0 in range(0, len(skipFit)):
        maskFit = (inds2D % skipFit[a0]) == 0;
        coefsInit = curve_fit(fit_func,
                              basis[np.logical_and(mask.reshape(-1), maskFit.reshape(-1))],
                              image[np.logical_and(mask, maskFit)],
                              p0=coefsInit,
                              bounds=(lb, ub)
                              )[0]
    A = 1.0
    B = coefsInit[3]
    C = coefsInit[2]
    coefs = [coefsInit[1], coefsInit[0], A, C, B]

    ellipse_rs = [coefs, coefsInit]
    return ellipse_rs


def polar_transformation_matlab(image, ellipse_rs, mask=None, **kargs):
    pixelSize = kargs.get('pixelSize', 1)
    rSigma = kargs.get('rSigma', 0.1)
    tSigma = kargs.get('tSigma', 1)
    rMax = kargs.get('rMax', 240)
    dr = kargs.get('dr', 2)
    dt = kargs.get('dt', 5)

    polarRadius = np.arange(0, rMax, dr) * pixelSize;
    polarTheta = np.arange(0, 360, dt) * (np.pi / 180)
    ya, xa = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))

    A, B, C = ellipse_rs[0][2], ellipse_rs[0][3], ellipse_rs[0][4]
    A, B, C = A/A, B/A, C/A
    coefs = [ellipse_rs[0][0], ellipse_rs[0][1], C, B]
    xa = xa - coefs[0]
    ya = ya - coefs[1]
    # Correction factors
    if abs(C) > -6:
        p0 = -np.arctan((1 - B + np.sqrt((B - 1) ** 2 + C ** 2)) / C);
    else:
        p0 = 0;

    a0 = np.sqrt(2 * (1 + C + np.sqrt((C - 1) ** 2 + B ** 2)) / (4 * C - B ** 2))
    b0 = np.sqrt(2 * (1 + C - np.sqrt((C - 1) ** 2 + B ** 2)) / (4 * C - B ** 2))
    ratio = b0 / a0;
    m = [[ratio * np.cos(p0) ** 2 + np.sin(p0) ** 2,
          -np.cos(p0) * np.sin(p0) + ratio * np.cos(p0) * np.sin(p0)],
         [-np.cos(p0) * np.sin(p0) + ratio * np.cos(p0) * np.sin(p0),
          np.cos(p0) ** 2 + ratio * np.sin(p0) ** 2]]
    m = np.array(m)
    ta = np.arctan2(m[1, 0] * xa + m[1, 1] * ya, m[0, 0] * xa + m[0, 1] * ya)
    ra = np.sqrt(xa ** 2 + coefs[2] * ya ** 2 + coefs[3] * xa * ya) * b0

    # Resamping coordinates
    Nout = [len(polarRadius), len(polarTheta)];
    rInd = (np.round((ra - polarRadius[0]) / dr)).astype(np.uint)
    tInd = (np.mod(np.round((ta - polarTheta[0]) / (dt * np.pi / 180)), Nout[1])).astype(np.uint)
    sub = np.logical_and(rInd <= Nout[0] - 1, rInd >= 0)
    rtIndsSub = np.array([rInd[sub], tInd[sub]]).T;

    polarNorm = accum(rtIndsSub, np.ones(np.sum(sub)))
    polarImg = accum(rtIndsSub, image[sub])
    polarImg = polarImg / polarNorm
    if mask is None:
        mask = np.ones(image.shape)
    polarMask = accum(rtIndsSub, mask[sub]) == 0

    polarImg = np.ma.masked_array(polarImg, polarMask)
    return polarImg


if __name__ == "__main__":
    from datacube import cube
    fp = "C:\\Users\\vlftj\\Documents\\sample41_Ta_AD\\Camera 230 mm Ceta 20201030 1709 0001_1_5s_1f_area01.mrc"
    dc = cube.PDFCube(fp, filetype='image')
    dc.mask = np.loadtxt('../assets/mask_data.txt', delimiter=',', dtype=np.bool)
    # dc.find_center()
    # dc.calculate_azimuthal_average()
    dc.center = [1201, 1073]
    polar_img = polar_transformation_matlab(dc.img_display, [[1073, 1201, 1, 0, 1]])
    plt.imshow(polar_img)
    plt.show()