import numpy as np
import xarray as xr

def xarray_to_endog_exog(ds: xr.Dataset,
                         endog_var: str,
                         include_space_coords: bool = False,
                         include_time_coords: bool = False
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Transform an xarray Dataset into NumPy arrays: endog and exog, plus metadata.

    Reshapes data from grid dimensions ('time', 'lat', 'lon') into ('time', 'space')
    or ('time', 'covariates', 'space') formats suitable for some modeling approaches.

    Parameters:
    -----------
    ds : xr.Dataset
        The input xarray Dataset with dimensions 'time', 'lat', 'lon' and
        data variables that may depend on one or more of these dimensions.
    endog_var : str
        The name of the data variable to be used as the endogenous variable.
        Must have dimensions ('time', 'lat', 'lon').
    include_space_coords : bool, optional (default=True)
        Whether to include 'lat' and 'lon' coordinates as covariates in the exog array.
    include_time_coords : bool, optional (default=False)
        Whether to include the 'time' coordinate as a covariate in the exog array.
        If True, the time coordinate value for each time step is repeated across
        all spatial locations for that step.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, dict]
        - endog: NumPy array of shape (t, s) representing the endogenous variable.
                 s = n_lat * n_lon is the total number of spatial locations.
        - exog: NumPy array of shape (t, c, s) representing the exogenous variables.
                c is the number of covariates.
        - metadata: Dictionary with keys:
            - 't': Number of time steps.
            - 'n_lat': Number of latitude points.
            - 'n_lon': Number of longitude points.
            - 's': Total number of spatial points (n_lat * n_lon).
            - 'covariates': List of dicts, each with 'name', 'original_dims',
                            and 'is_coord' for each exog covariate.

    Raises:
    -------
    ValueError
        If 'endog_var' is not found or does not have the required dimensions ('time', 'lat', 'lon').
        If a data variable has unsupported dimensions.
        If no exogenous variables are generated.
    KeyError
        If required dimensions 'time', 'lat', or 'lon' are missing from the dataset.
    """
    # Extract dimension sizes and validate required dimensions
    try:
        t = len(ds['time'])
        n_lat = len(ds['lat'])
        n_lon = len(ds['lon'])
    except KeyError as e:
        raise KeyError(f"Dataset is missing required dimension: {e}. Expected 'time', 'lat', 'lon'.")

    s = n_lat * n_lon  # Total number of spatial locations

    # Validate endog_var
    if endog_var not in ds.data_vars:
        raise ValueError(f"Endogenous variable '{endog_var}' not found in dataset.")
    if set(ds[endog_var].dims) != {'time', 'lat', 'lon'}:
        raise ValueError(f"Endogenous variable '{endog_var}' must have dimensions ('time', 'lat', 'lon'). Found {ds[endog_var].dims}")

    # Extract endog: stack spatial dimensions
    # Ensure the stacking order ('lat', 'lon') matches flattening below
    endog = ds[endog_var].stack(space=('lat', 'lon')).values  # Shape: (t, s)

    # Initialize list for exog covariates and metadata
    exog_list = []
    covariates_metadata = []

    # Include spatial covariates (lat and lon) if requested
    if include_space_coords:
        # Create meshgrid - 'ij' indexing ensures shape (n_lat, n_lon)
        lat_mesh, lon_mesh = np.meshgrid(ds['lat'].values, ds['lon'].values, indexing='ij')
        # Flatten in C-order (row-major), consistent with stack('lat', 'lon')
        lat_flat = lat_mesh.flatten()  # Shape: (s,)
        lon_flat = lon_mesh.flatten()  # Shape: (s,)
        # Repeat the spatial pattern for each time step
        lat_repeated = np.repeat(lat_flat[None, :], t, axis=0)  # Shape: (t, s)
        lon_repeated = np.repeat(lon_flat[None, :], t, axis=0)  # Shape: (t, s)

        exog_list.extend([lat_repeated, lon_repeated])
        covariates_metadata.append({'name': 'lat', 'original_dims': ('lat',), 'is_coord': True})
        covariates_metadata.append({'name': 'lon', 'original_dims': ('lon',), 'is_coord': True})

    # Include time coordinate if requested
    if include_time_coords:
        time_vals = ds['time'].values # Shape: (t,)
        # Ensure time_vals are numerical if they are datetime/timedelta
        if np.issubdtype(time_vals.dtype, np.datetime64) or np.issubdtype(time_vals.dtype, np.timedelta64):
             # Example: Convert to nanoseconds since epoch, then to float seconds
             # Adjust conversion as needed for your specific modeling context
             print("Warning: Time coordinate appears to be datetime/timedelta. Converting to float seconds since epoch.")
             time_vals = time_vals.astype(np.int64) / 1e9

        # Repeat the time value across all spatial points for each time step
        time_repeated = np.repeat(time_vals[:, None], s, axis=1) # Shape: (t, s)

        exog_list.append(time_repeated)
        covariates_metadata.append({'name': 'time', 'original_dims': ('time',), 'is_coord': True})

    # Process each data variable (potential exogenous covariates) except endog_var
    for var in ds.data_vars:
        if var == endog_var:
            continue

        var_data = ds[var]
        dims = var_data.dims
        data = var_data.values
        original_dims_tuple = tuple(dims) # Store original dims for metadata

        # Reshape/broadcast based on original dimensions
        if set(dims) == {'time', 'lat', 'lon'}:
            # Already has all dims, just stack space
            var_stacked = var_data.stack(space=('lat', 'lon')).values  # Shape: (t, s)
            exog_list.append(var_stacked)
        elif set(dims) == {'lat', 'lon'}:
            # Spatial only, flatten and repeat over time
            var_flat = data.reshape(s) # Flatten assuming 'lat' then 'lon' order
            var_broadcast = np.repeat(var_flat[None, :], t, axis=0)  # Shape: (t, s)
            exog_list.append(var_broadcast)
        elif len(dims) == 1:
            # Single dimension, broadcast appropriately
            if dims[0] == 'time':
                # Time only, repeat over space
                var_broadcast = np.repeat(data[:, None], s, axis=1)  # Shape: (t, s)
                exog_list.append(var_broadcast)
            elif dims[0] == 'lat':
                # 'lat' only, tile across 'lon', flatten, repeat over time
                var_tiled = np.tile(data[:, None], (1, n_lon)).flatten() # Shape: (s,)
                var_broadcast = np.repeat(var_tiled[None, :], t, axis=0) # Shape: (t, s)
                exog_list.append(var_broadcast)
            elif dims[0] == 'lon':
                 # 'lon' only, tile across 'lat', flatten, repeat over time
                 var_tiled = np.tile(data[None, :], (n_lat, 1)).flatten() # Shape: (s,)
                 var_broadcast = np.repeat(var_tiled[None, :], t, axis=0) # Shape: (t, s)
                 exog_list.append(var_broadcast)
            else:
                 # Unsupported single dimension name
                 raise ValueError(f"Variable '{var}' has unsupported single dimension: {dims[0]}. Expected 'time', 'lat', or 'lon'.")
        elif len(dims) == 0: # Scalar variable
            # Broadcast scalar to all points in time and space
            var_broadcast = np.full((t, s), data.item()) # Shape: (t, s)
            exog_list.append(var_broadcast)
        else:
            # Any other dimension combination is unsupported
            raise ValueError(f"Variable '{var}' has unsupported dimensions: {dims}. Expected combinations of ('time', 'lat', 'lon') or scalar.")

        # Add metadata for this processed variable if it was added
        if var not in [m['name'] for m in covariates_metadata]: # Avoid duplicates if somehow processed differently
             covariates_metadata.append({'name': var, 'original_dims': original_dims_tuple, 'is_coord': False})


    # Stack all prepared exog covariates into the final exog array
    if not exog_list:
        # Raise error only if *no* covariates were generated at all
        raise ValueError("No exogenous variables generated. Ensure include_space_coords/include_time_coords is True or add suitable data variables.")

    # Stack along a new 'covariate' dimension (axis=1)
    exog = np.stack(exog_list, axis=1)  # Shape: (t, c, s)

    # Compile final metadata
    metadata = {
        't': t,
        'n_lat': n_lat,
        'n_lon': n_lon,
        's': s,
        'covariates': covariates_metadata
    }

    return endog, exog, metadata

