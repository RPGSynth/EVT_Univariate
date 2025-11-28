import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import warnings
from typing import Union, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .gev_results import GEVFit

class GEVPlotter:
    """
    Visualization accessor for GEVFit objects.
    Handles all Matplotlib logic, separating UI from statistical computations.
    """
    def __init__(self, fit_obj: 'GEVFit'):
        self.fit = fit_obj

    def return_levels(self,
                  T: Union[List[float], np.ndarray],
                  t: Union[int, List[int], None] = None,
                  s: int = 0,
                  show_ci: bool = True,
                  ax: Optional[plt.Axes] = None,
                    **kwargs):
        """
        Parameters
        ----------
        T : list or np.ndarray
            Return periods.
        t : int, list of int, or None
            Time indices. If None, uses [0, mid, last].
        s : int
            Site index (single site only).
        show_ci : bool
            Whether to show confidence bands returned by ReturnLevel.compute().
        ax : plt.Axes, optional
            Axis to plot on. If None, a new figure/axis is created.
        **kwargs :
            Passed to ax.plot() (e.g. linestyle, alpha, etc.).
            Optional:
                - color / c : force a single color for all lines
                - cmap       : colormap for multiple lines
        """

        # --- 1. SAFETY CHECK FOR S ---
        if np.size(s) > 1:
            raise ValueError(
                "Plotting multiple sites simultaneously is ambiguous. "
                "Please provide a single integer for 's' (Space). "
                "To compare sites, call .plot.return_levels() multiple times on the same axis."
            )

        # 2. Time indices
        if t is None:
            n_obs = self.fit.n_obs
            t_indices = sorted(list(set([0, n_obs // 2, n_obs - 1])))
        else:
            t_indices = np.atleast_1d(t).tolist()

        T = np.asarray(T, dtype=float)

        # 3. Check for Singularity (Polite Warning)
        if np.any(T <= 1.0):
            warnings.warn(
                "Input 'T' contains values <= 1.0. These imply probability=0 (log(-inf)) and will be hidden.",
                UserWarning
            )

        # 4. Setup Canvas
        if ax is None:
            figsize = kwargs.pop('figsize', (10, 6))
            fig, ax = plt.subplots(figsize=figsize)

        # --- 5. COLOR LOGIC ---
        user_color = kwargs.get('color', kwargs.get('c', None))
        user_cmap = kwargs.pop('cmap', None)

        if user_color is not None:
            final_colors = [user_color] * len(t_indices)
        elif user_cmap is not None:
            final_colors = user_cmap(np.linspace(0, 1, len(t_indices)))
        else:
            if len(t_indices) > 1:
                final_colors = plt.cm.magma(np.linspace(0.2, 0.85, len(t_indices)))
            else:
                final_colors = ['#1f77b4']

        # Clean kwargs for ax.plot
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in ['color', 'c', 'label']}

        # --- 6. SMART USE OF ReturnLevel.compute ---
        # Build one ReturnLevel over all requested times and this single site
        rl_obj = self.fit.return_level(t=t_indices, s=s)

        # compute now returns: zp, se, ci
        # shapes: (N_t, N_s, N_periods) and (N_t, N_s, N_periods, 2)
        levels, ses, ci = rl_obj.compute(T)

        # Since s is scalar, N_s == 1; we’ll always index [:, 0, :]
        # Main plotting loop over time index axis
        for i, t_idx in enumerate(t_indices):
            y = levels[i, 0, :]       # (N_periods,)
            y_se = ses[i, 0, :]       # (N_periods,)
            y_ci = ci[i, 0, :, :]     # (N_periods, 2)

            # Mask: finite only (will naturally drop bad T or NaNs)
            mask = np.isfinite(y) & np.isfinite(y_se)
            if not np.any(mask):
                continue

            T_valid = T[mask]
            y_valid = y[mask]
            se_valid = y_se[mask]
            ci_valid = y_ci[mask, :]  # (N_valid, 2)

            lower_valid = ci_valid[:, 0]
            upper_valid = ci_valid[:, 1]

            # Label for legend
            label = f"t={t_idx}"
            if s > 0:
                label += f", s={s}"

            color_to_use = final_colors[i]

            # Plot line
            line, = ax.plot(
                T_valid,
                y_valid,
                label=label,
                color=color_to_use,
                lw=2.5,
                **plot_kwargs,
            )

            # Plot CI band from compute()
            if show_ci:
                ax.fill_between(
                    T_valid,
                    lower_valid,
                    upper_valid,
                    color=line.get_color(),
                    alpha=0.15,
                    lw=0,
                )

        # 7. AESTHETICS
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(True, which="major", ls="-", alpha=0.2, color='k')
        ax.grid(True, which="minor", ls=":", alpha=0.1, color='k')

        ax.set_title(f"Return Level Plot (Site {s})", fontsize=12, pad=15)
        ax.set_xlabel("Return Period (Years)", fontsize=10)
        ax.set_ylabel("Return Level", fontsize=10)

        if len(t_indices) > 0:
            ax.legend(frameon=False, loc='best')

        return ax
    
    
    def spatial_map(self, 
                    T: float, 
                    t: int = 0, 
                    coordinates: Optional[tuple] = None, 
                    grid_shape: Optional[tuple] = None,
                    s_mask: Optional[np.ndarray] = None,
                    ax: Optional[plt.Axes] = None,
                    **kwargs):
        """
        Plots a spatial map of Return Levels for a fixed time and return period.
        
        Can render as a Scatter plot (if coordinates provided) or Heatmap (if grid_shape provided).

        Args:
            T (float): Single Return Period (e.g., 100).
            t (int): Single time index to evaluate.
            coordinates (tuple): (x_array, y_array) for scattered stations (Long, Lat).
            grid_shape (tuple): (rows, cols) to reshape the flattened sites into a grid.
            s_mask (array): Optional boolean mask to plot only a subset of sites.
            ax (plt.Axes): Matplotlib axes.
            **kwargs: Arguments passed to scatter() or imshow() (e.g., cmap, vmin, vmax).
        """
        
        # 1. Validation: We need exactly ONE definition of space
        if coordinates is None and grid_shape is None:
            raise ValueError("You must provide either 'coordinates=(x, y)' or 'grid_shape=(rows, cols)' to map the data.")
        if coordinates is not None and grid_shape is not None:
            raise ValueError("Provide 'coordinates' OR 'grid_shape', not both.")

        # 2. Setup S (All sites by default)
        n_samples = self.fit.data.n_samples
        s_indices = np.arange(n_samples)
        
        # 3. Compute (Using the Factory)
        # We compute for ALL sites at once.
        rl_obj = self.fit.return_level(t=t, s=s_indices)
        
        # Result shape: (1, S, 1) because t=scalar, s=vector, T=scalar
        levels, _, _ = rl_obj.compute([T])
        
        # Flatten to (S,)
        values = levels.flatten()

        # 4. Apply subset mask if provided (e.g., masking oceans)
        if s_mask is not None:
            values = values[s_mask]
            # Note: coordinates handling would need masking too, handled below

        # 5. Setup Canvas
        if ax is None:
            figsize = kwargs.pop('figsize', (8, 6))
            fig, ax = plt.subplots(figsize=figsize)

        # 6. PLOT LOGIC
        
        # --- OPTION A: SCATTER MAP (Stations) ---
        if coordinates is not None:
            x, y = coordinates
            if s_mask is not None:
                x, y = x[s_mask], y[s_mask]
                
            # Scatter plot
            # Default to 'viridis' or 'plasma' if no cmap given
            cmap = kwargs.pop('cmap', 'plasma')
            s_size = kwargs.pop('s', 20) # Dot size
            
            mappable = ax.scatter(x, y, c=values, cmap=cmap, s=s_size, **kwargs)
            ax.set_aspect('equal') # Important for maps
            ax.set_title(f"{int(T)}-Year Return Level (t={t})")

        # --- OPTION B: GRID MAP (Heatmap) ---
        elif grid_shape is not None:
            # Check shapes
            if np.prod(grid_shape) != len(values):
                raise ValueError(f"Grid shape {grid_shape} size {np.prod(grid_shape)} != N_sites {len(values)}")
            
            # Reshape (Deflatten)
            grid_values = values.reshape(grid_shape)
            
            cmap = kwargs.pop('cmap', 'plasma')
            
            # Imshow with origin='lower' is standard for maps
            mappable = ax.imshow(grid_values, origin='lower', cmap=cmap, **kwargs)
            ax.set_title(f"{int(T)}-Year Return Level (t={t})")

        # 7. Add Colorbar
        plt.colorbar(mappable, ax=ax, label="Return Level")
        
        return ax
    
    def diagnostics(self):
        """Placeholder for QQ plots."""
        print("Coming soon...")