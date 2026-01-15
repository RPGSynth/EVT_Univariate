import numpy as np
from scipy.stats import norm
import pandas as pd
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .gev_rlevels import ReturnLevel
    from .gev_plots import GEVPlotter

class GEVFit:
    """
    Container for the results of a GEV optimization.
    Handles AIC/BIC, Confidence Intervals, and Summary formatting.
    """
    def __init__(self, 
                 params: np.ndarray, 
                 cov_matrix: np.ndarray, 
                 nll_avg: float, 
                 data, 
                 linker, 
                 confidence,
                 reparam_T: Optional[float] = None,
                ):
        
        self.params = np.asarray(params)
        self.cov_matrix = np.asarray(cov_matrix)
        
        # Safe Standard Errors (handle negative diagonals if numerical issues occur)
        diag_cov = np.diag(self.cov_matrix)
        safe_diag = np.where(diag_cov >= 0, diag_cov, np.nan)
        self.se = np.sqrt(safe_diag)
        self.confidence = float(confidence)
        
        self.nll_avg = float(nll_avg)
        self.data = data
        self.linker = linker
        self.reparam_T = reparam_T
        
        self.dims = self.data.covariate_dims
        self.n_obs = data.n_obs

    @property
    def n_params(self):
        return self.params.size
    
    @property
    def nll_total(self):
        W_total = np.sum(self.data.weights)
        return self.nll_avg * W_total
    
    @property
    def aic(self):
        return 2 * self.n_params + 2 * self.nll_total
    
    @property
    def bic(self):
        return self.n_params * np.log(self.data.n_obs) + 2 * self.nll_total
    
  
    def ci(self, confidence: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the Wald Confidence Intervals for the parameters.
        
        Args:
            confidence: The confidence level (e.g., 0.95). 
                        If None, uses the class default.
        
        Returns:
            (lower_bounds, upper_bounds): Tuple of numpy arrays.
        """
        if confidence is None:
            confidence = self.confidence
            
        alpha = 1.0 - confidence
        # Two-tailed critical value (e.g., 1.96 for 95%)
        z_crit = norm.ppf(1.0 - alpha / 2.0)
        
        lower = self.params - z_crit * self.se
        upper = self.params + z_crit * self.se
        
        return lower, upper

    def to_dataframe(self, confidence: Optional[float] = None) -> pd.DataFrame:
        """
        Returns the coefficients table as a Pandas DataFrame.
        """
        # Determine level
        if confidence is None:
            confidence = self.confidence
            
        # Call the math method
        ci_lower, ci_upper = self.ci(confidence)
        
        # Generate Names
        names = []
        name_1 = f"Zp({int(self.reparam_T)})" if self.reparam_T else "Loc"
        param_groups = [name_1, 'Scale', 'Shape']
        
        for prefix, dim in zip(param_groups, self.dims):
            names.extend([f"{prefix}_{i}" for i in range(dim)])
            
        # Calculate other stats
        z_scores = self.params / self.se
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        
        # Dynamic Headers
        alpha = 1.0 - confidence
        col_lower = f"[{alpha/2:.3f}"
        col_upper = f"{1-alpha/2:.3f}]"
        
        return pd.DataFrame({
            'Estimate': self.params,
            'Std.Err': self.se,
            'Z-score': z_scores,
            'P-value': p_values,
            col_lower: ci_lower,
            col_upper: ci_upper
        }, index=names)

    def __str__(self):
        """Generates the summary string representation."""
        df = self.to_dataframe()
        
        width = 78
        header_lines = [
            "\n" + "="*width,
            f"{'GEV FIT RESULTS (JAX Engine)':^{width}}",
            "="*width,
            f"Nobs: {self.n_obs:<10} | Nsamples: {self.data.n_samples:<10}",
            f"NLL: {self.nll_total:.2f}",
            f"Average NLL: {self.nll_avg:.2f}",
            f"AIC: {self.aic:.2f}        | BIC: {self.bic:.2f}",
            f"Confidence Level: {self.confidence*100:.0f}%"
        ]
        
        if self.reparam_T:
            header_lines.append(f"\n*Note: Location (μ) reparameterized as {int(self.reparam_T)}-Year Return Level (Zp).")
            
        header_lines.append("-" * width)
        
        def signi(val):
            if np.isnan(val): return "   "
            if val < 0.001: return "***"
            if val < 0.01: return "** "
            if val < 0.05: return "* "
            return "   "

        # Extract dynamic column names
        cols = df.columns
        ci_low_name, ci_high_name = cols[4], cols[5]
        
        col_str = f"{'Param':<15} {'Estimate':<10} {'Std.Err':<10} {'Z':<8} {'P>|z|':<10} {ci_low_name:<10} {ci_high_name:<10}"
        header_lines.append(col_str)
        header_lines.append("-" * width)
        
        rows = []
        for name, row in df.iterrows():
            stars = signi(row['P-value'])
            rows.append(
                f"{name:<15} {row['Estimate']:<10.4f} {row['Std.Err']:<10.4f} {row['Z-score']:<8.2f} {row['P-value']:<10.4f} {row[ci_low_name]:<10.4f} {row[ci_high_name]:<10.4f} {stars}"
            ) 
        
        footer = "-" * width
        return "\n".join(header_lines + rows + [footer])
    
    def summary(self):
        print(self.__str__())

    # --- Accessors ---
    def return_level(self, t=None, s=None):
        from .gev_rlevels import ReturnLevel 
        return ReturnLevel(self, t=t, s=s)
    
    @property
    def plot(self) -> 'GEVPlotter':
        from .gev_plots import GEVPlotter 
        return GEVPlotter(self)