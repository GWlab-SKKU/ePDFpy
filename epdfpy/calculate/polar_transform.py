import numpy as np

def cartesian_to_polarelliptical_transform(
        cartesianData,
        p_ellipse,
        dr=1,
        dphi=np.radians(2),
        r_range=None,
        mask=None,
        maskThresh=0.99,
):
    """
    Transforms an array of data in cartesian coordinates into a data array in
    polar-elliptical coordinates.
    Discussion of the elliptical parametrization used can be found in the docstring
    for the process.utils.elliptical_coords module.
    Args:
        cartesianData (2D float array): the data in cartesian coordinates
        p_ellipse (5-tuple): specifies (qx0,qy0,a,b,theta), the parameters for the
            transformation. These are the same 5 parameters which are outputs
            of the elliptical fitting functions in the process.calibration
            module, e.g. fit_ellipse_amorphous_ring and fit_ellipse_1D. For
            more details, see the process.utils.elliptical_coords module docstring
        dr (float): sampling of the (r,phi) coords: the width of the bins in r
        dphi (float): sampling of the (r,phi) coords: the width of the bins in phi,
            in radians
        r_range (number or length 2 list/tuple or None): specifies the sampling of the
            (r,theta) coords.  Precise behavior which depends on the parameter type:
                * if None, autoselects max r value
                * if r_range is a number, specifies the maximum r value
                * if r_range is a length 2 list/tuple, specifies the min/max r values
        mask (2d array of bools): shape must match cartesianData; where mask==False,
            ignore these datapoints in making the polarElliptical data array
        maskThresh (float): the final data mask is calculated by converting mask (above)
            from cartesian to polar elliptical coords.  Due to interpolation, this
            results in some non-boolean values - this is converted back to a boolean
            array by taking polarEllipticalMask = polarTrans(mask) < maskThresh. Cells
            where polarTrans is less than 1 (i.e. has at least one masked NN) should
            generally be masked, hence the default value of 0.99.
    Returns:
        (3-tuple): A 3-tuple, containing:
            * **polarEllipticalData**: *(2D masked array)* a masked array containing
              the data and the data mask, in polarElliptical coordinates
            * **rr**: *(2D array)* meshgrid of the r coordinates
            * **pp**: *(2D array)* meshgrid of the phi coordinates
    """
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
    # p_bins = np.arange(-np.pi + dphi / 2.0, np.pi + dphi / 2.0, dphi)  # Change from -180 ~ 180 -> 0 ~ 360 => Angle start from 0
    p_bins = np.arange(dphi / 2.0, 2*np.pi + dphi / 2.0, dphi)
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
