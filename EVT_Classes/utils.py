# --- Contents of utils.py ---
import xarray as xr
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Optional, List, Union

import xarray as xr
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Optional, List, Union

def xarray_to_gev(
    endog_da: xr.DataArray,
    exog_ds: Optional[xr.Dataset] = None,
    time_dim: str = 'time',
    spatial_dims: Optional[List[str]] = None,
    dims_as_covariates: Optional[List[str]] = None
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    (Docstring is the same as before)
    """
    if exog_ds is not None:
        exog_ds = exog_ds.copy()
    else:
        exog_ds = xr.Dataset(coords=endog_da.coords)

    if dims_as_covariates:
        vars_to_add = {}
        for dim_name in dims_as_covariates:
            if dim_name in endog_da.coords:
                coord_as_var = endog_da[dim_name]

                # If it's a date, convert it to a number. The name is handled below.
                if np.issubdtype(coord_as_var.dtype, np.datetime64):
                    coord_as_var = (coord_as_var - coord_as_var[0]).dt.days + 1
                
                # --- CHANGE IS HERE ---
                # Standardize the new variable name to always use a '_cov' suffix.
                new_var_name = f"{dim_name}_cov"
                # --- END OF CHANGE ---
                
                if new_var_name in exog_ds.data_vars:
                    warnings.warn(f"Covariate '{new_var_name}' already exists.")
                else:
                    broadcasted_coord, _ = xr.broadcast(coord_as_var, endog_da)
                    vars_to_add[new_var_name] = broadcasted_coord
        
        if vars_to_add:
            exog_ds = exog_ds.assign(**vars_to_add)

    if spatial_dims is None:
        spatial_dims = [str(dim) for dim in endog_da.dims if dim != time_dim]

    stacked_endog = endog_da.stack(space=spatial_dims).transpose(time_dim, 'space')
    space_coords_df = stacked_endog.coords['space'].variable.to_index().to_frame(index=False)
    endog_np = stacked_endog.values
    
    results = { 'endog': endog_np, 'spatial_coords': space_coords_df }

    if exog_ds and len(exog_ds.data_vars) > 0:
        # We need the covariate names for the inverse function. Let's add them.
        results['covariate_names'] = list(exog_ds.data_vars)
        stacked_exog_ds = exog_ds.stack(space=spatial_dims)
        exog_da = stacked_exog_ds.to_array(dim='covariate')
        exog_np = exog_da.transpose(time_dim, 'covariate', 'space').values
        results['exog'] = exog_np
            
    return results

def gev_to_xarray(
    endog_flat: np.ndarray,
    spatial_coords: pd.DataFrame,
    exog_flat: Optional[np.ndarray] = None,
    time_coords: Optional[Union[np.ndarray, pd.Index]] = None,
    covariate_names: Optional[List[str]] = None,
    time_dim_name: str = "time",
) -> Dict[str, Union[xr.DataArray, xr.Dataset]]:
    """
    (Docstring is the same as before)
    """
    # 1. Prepare coordinates
    if time_coords is None:
        time_coords = np.arange(endog_flat.shape[0])
    if len(time_coords) != endog_flat.shape[0]:
        raise ValueError("Length of time_coords does not match time dimension of endog_flat.")
    spatial_dim_names = spatial_coords.columns.tolist()
    spatial_multi_index = pd.MultiIndex.from_frame(spatial_coords, names=spatial_dim_names)

    # 2. Reconstruct the endogenous DataArray
    endog_da_flat = xr.DataArray(
        data=endog_flat,
        dims=[time_dim_name, "space"],
        coords={time_dim_name: time_coords, "space": spatial_multi_index},
    )
    endog_da = endog_da_flat.unstack("space")
    results = {"endog_da": endog_da}

    # 3. Reconstruct the exogenous Dataset (if provided)
    if exog_flat is not None:
        if covariate_names is None:
             raise ValueError("`covariate_names` must be provided to reconstruct the exogenous dataset.")
        
        num_covariates = exog_flat.shape[1]
        if len(covariate_names) != num_covariates:
            raise ValueError(f"Provided {len(covariate_names)} covariate names but exog_flat has {num_covariates} covariates.")

        exog_da_flat = xr.DataArray(
            data=exog_flat,
            dims=[time_dim_name, "covariate", "space"],
            coords={
                time_dim_name: time_coords,
                "covariate": covariate_names,
                "space": spatial_multi_index,
            },
        )
        exog_da_unstacked = exog_da_flat.unstack("space")
        
        # --- CHANGE IS HERE ---
        # Filter out helper covariates (ending in '_cov') before creating the Dataset
        original_cov_names = [
            name for name in covariate_names if not name.endswith("_cov")
        ]
        
        exog_ds = xr.Dataset()
        for cov_name in original_cov_names:
            exog_ds[cov_name] = exog_da_unstacked.sel(covariate=cov_name, drop=True)
        # --- END OF CHANGE ---

        results["exog_ds"] = exog_ds

    return results


def xarray_to_endog_exog(ds: xr.Dataset,
                         endog_var: str,
                         include_space_coords: bool = True,
                         include_time_coords: bool = True
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


def endog_exog_to_xarray(endog: np.ndarray, exog: np.ndarray, endog_var: str, metadata: dict) -> xr.Dataset:
    """
    Transform endog and exog NumPy arrays back into an xarray Dataset using metadata.

    Parameters:
    -----------
    endog : np.ndarray
        The endogenous variable array of shape (t, s).
    exog : np.ndarray
        The exogenous variables array of shape (t, c, s).
    endog_var : str
        The name of the endogenous variable.
    metadata : dict
        Metadata from xarray_to_endog_exog with 't', 'n_lat', 'n_lon', and 'covariates'.

    Returns:
    --------
    xr.Dataset
        The reconstructed xarray Dataset with dimensions 'time', 'lat', 'lon'.

    Raises:
    -------
    ValueError
        If the shapes of endog or exog do not match the metadata dimensions.
        If the number of covariates in exog does not match metadata.
    """
    # Extract metadata
    t = metadata['t']
    n_lat = metadata['n_lat']
    n_lon = metadata['n_lon']
    s = n_lat * n_lon
    covariates = metadata['covariates']

    # Validate input shapes
    if endog.shape != (t, s):
        raise ValueError(f"endog must have shape ({t}, {s}), got {endog.shape}")
    if exog.shape != (t, len(covariates), s):
        raise ValueError(f"exog must have shape ({t}, {len(covariates)}, {s}), got {exog.shape}")

    # Initialize coordinate arrays
    time = np.arange(t)
    lat = None
    lon = None

    # Extract lat and lon from exog where is_coord=True
    for i, cov in enumerate(covariates):
        if cov['is_coord']:
            data = exog[:, i, :]
            if cov['name'] == 'lat':
                # Take the first time slice and reshape to original lat array
                lat = data[0, :].reshape(n_lat, n_lon)[:, 0]  # Extract unique lat values
            elif cov['name'] == 'lon':
                # Take the first time slice and reshape to original lon array
                lon = data[0, :].reshape(n_lat, n_lon)[0, :]  # Extract unique lon values

    if lat is None or lon is None:
        raise ValueError("Latitude or longitude coordinates not found in exog covariates.")

    # Reshape endog back to (t, n_lat, n_lon)
    endog_reshaped = endog.reshape(t, n_lat, n_lon)

    # Initialize the dataset with the endogenous variable and original coordinates
    ds = xr.Dataset(
        {endog_var: (['time', 'lat', 'lon'], endog_reshaped)},
        coords={'time': time, 'lat': lat, 'lon': lon}
    )

    # Reconstruct exogenous variables (excluding coordinates)
    for i, cov in enumerate(covariates):
        if cov['is_coord']:
            continue  # Skip lat and lon as they’re already in coords
        name = cov['name']
        dims = cov['dims']
        data = exog[:, i, :]

        if set(dims) == {'time', 'lat', 'lon'}:
            ds[name] = (['time', 'lat', 'lon'], data.reshape(t, n_lat, n_lon))
        elif set(dims) == {'lat', 'lon'}:
            ds[name] = (['lat', 'lon'], data[0].reshape(n_lat, n_lon))
        elif len(dims) == 1:
            if dims[0] == 'time':
                ds[name] = (['time'], data[:, 0])
            elif dims[0] == 'lat':
                ds[name] = (['lat'], data[0].reshape(n_lat, n_lon).mean(axis=1))
            elif dims[0] == 'lon':
                ds[name] = (['lon'], data[0].reshape(n_lat, n_lon).mean(axis=0))
        else:
            raise ValueError(f"Unsupported dimensions for variable '{name}': {dims}")
            
    return ds
