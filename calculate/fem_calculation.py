
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
from scipy.optimize import leastsq
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
    _a,_b,_theta = convert_ellipse_params(_A,_B,_C)
    return (_x0,_y0,_a,_b,_theta),p

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
