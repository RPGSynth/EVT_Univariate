import numpy as np
from scipy.stats import norm
import pandas as pd
from .gev_rlevels import ReturnLevel

class GEVFit:
    """
    Container for the results of a GEV optimization.
    Replaces the old 'GEVFit' class, handling AIC/BIC and Summary formatting.
    """
    def __init__(self, params, cov_matrix, n_ll_avg, input_data, linker,reparam_T=None):
        self.params = np.asarray(params)
        self.cov_matrix = np.asarray(cov_matrix)
        diag_cov = np.diag(self.cov_matrix)
        safe_diag = np.where(diag_cov >= 0, diag_cov, np.nan)
        self.se = np.sqrt(safe_diag)
        self.n_ll_avg = float(n_ll_avg)
        self.input = input_data
        self.linker = linker
        self.reparam_T = reparam_T
        
        # Dimensions derived from input data
        self.dims = self.input.covariate_dims
        self.n_obs = input_data.n_obs

    @property
    def n_params(self):
        return self.params.size
    
    @property
    def n_ll_total(self):
        """Recover total NLL from the average."""
        W_total = np.sum(self.input.weights)
        return self.n_ll_avg * W_total
    
    @property
    def aic(self):
        return 2 * self.n_params + 2 * self.n_ll_total
    
    @property
    def bic(self):
        return self.n_params * np.log(self.input.n_obs) + 2 * self.n_ll_total
    
    def return_level(self, t=None, s=None):
        """
        Factory method to create a ReturnLevel object for this fit.
        """
        # Lazy import to avoid circular dependency at top of file
        from .gev_rlevels import ReturnLevel 
        return ReturnLevel(self, t=t, s=s)
    
    @property
    def plot(self):
        """
        Accessor for plotting methods.
        Usage: fit.plot.return_levels(...)
        """
        # Lazy import ensures matplotlib is only loaded when needed
        from .gev_plots import GEVPlotter 
        return GEVPlotter(self)

    def plot_return_levels(self, T, t=None, s=0, show_ci=True, ax=None):
        """
        API Wrapper to match the old GEVFit.plot_return_levels behavior.
        
        Args:
            T (array): Return periods.
            t (list[int]): List of time indices to plot curves for.
            s (int): Sample index (default 0).
        """
        import matplotlib.pyplot as plt
        
        if t is None:
            # Default to start, middle, end
            t = [0, self.input.n_obs // 2, self.input.n_obs - 1]
        
        if not isinstance(t, list):
            t = [t]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        # Generate colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(t)))

        for i, t_idx in enumerate(t):
            rl = ReturnLevel(self, t_idx=t_idx, s_idx=s)
            rl.plot_on_axis(T, ax=ax, color=colors[i], show_ci=show_ci, label_prefix=f"t={t_idx}")
            
        ax.set_title(f"Return Levels (Sample s={s})")
        ax.legend()
        return fig, ax
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns the coefficients table as a Pandas DataFrame.
        Handles dynamic naming for Reparameterization.
        """
        # 1. Generate Dynamic Parameter Names
        names = []
        
        # If reparam_T is set (e.g. 100), label first params as "Zp(100)"
        # Otherwise label as "Loc"
        name_1 = f"Zp({int(self.reparam_T)})" if self.reparam_T else "Loc"
        
        param_groups = [name_1, 'Scale', 'Shape']
        
        for prefix, dim in zip(param_groups, self.dims):
            names.extend([f"{prefix}_{i}" for i in range(dim)])
            
        # 2. Calculate Stats
        z_scores = self.params / self.se
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        
        # 95% CI
        z_crit = 1.96
        ci_lower = self.params - z_crit * self.se
        ci_upper = self.params + z_crit * self.se
        
        # 3. Build DataFrame
        df = pd.DataFrame({
            'Estimate': self.params,
            'Std.Err': self.se,
            'Z-score': z_scores,
            'P-value': p_values,
            '[0.025': ci_lower,
            '0.975]': ci_upper
        }, index=names)
        
        return df

    def __str__(self):
        """Generates the summary string representation."""
        df = self.to_dataframe()
        
        width = 78
        header_lines = [
            "\n" + "="*width,
            f"{'GEV FIT RESULTS (JAX Engine)':^{width}}",
            "="*width,
            f"Nobs: {self.n_obs:<10} | Nsamples: {self.input.n_samples:<10}",
            f"NLL: {self.n_ll_total:.2f}",
            f"Average NLL: {self.n_ll_avg:.2f}",
            f"AIC: {self.aic:.2f}        | BIC: {self.bic:.2f}",
        ]
        
        # --- THE REPARAM INFORMATION ---
        if self.reparam_T:
            header_lines.append(f"\n*Note: Location (μ) reparameterized as {int(self.reparam_T)}-Year Return Level (Zp).")
            
        header_lines.append("-" * width)
        
        # Helper to format p-values with stars
        def signi(val):
            if np.isnan(val): return "   "
            if val < 0.001: return "***"
            if val < 0.01: return "** "
            if val < 0.05: return "* "
            return "   "

        # Column Headers
        cols = f"{'Param':<15} {'Estimate':<10} {'Std.Err':<10} {'Z':<8} {'P>|z|':<10} {'[0.025':<10} {'0.975]':<10}"
        header_lines.append(cols)
        header_lines.append("-" * width)
        
        # Rows
        rows = []
        for name, row in df.iterrows():
            stars = signi(row['P-value'])
            rows.append(
                f"{name:<15} {row['Estimate']:<10.4f} {row['Std.Err']:<10.4f} {row['Z-score']:<8.2f} {row['P-value']:<10.4f} {row['[0.025']:<10.4f} {row['0.975]']:<10.4f} {stars}"
            ) 
        
        footer = "-" * width
        return "\n".join(header_lines + rows + [footer])
    
    def summary(self):
        """Prints the traditional summary table."""
        print(self.__str__())