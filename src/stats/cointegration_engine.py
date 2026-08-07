import pandas as pd
import numpy as np
import logging
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from dataclasses import dataclass, field
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass  
class CointegrationResult:
    """Result of a bivariate Engle-Granger Test"""
    is_cointegrated: bool
    p_value: float = 1.0
    test_statistic: float = np.nan
    critical_values: Dict[str, float] = field(default_factory=dict)
    hedge_ratio: float = np.nan
    intercept: float = np.nan
    spread_series: Optional[pd.Series] = None
    half_life: float = np.nan
    r_squared: float = np.nan
    dependent_var: str = ""
    independent_var: str = ""

class CointegrationAnalyzer:
    """
    Supports bivariate Engle-Granger (pairs) and N-asset Johansen (baskets),
    sharing a common half-life diagnostic across both.
    """

    def __init__(self, significance_level: float = 0.05, min_obs: int = 60):
        self.alpha = significance_level
        self.min_obs = min_obs

    @classmethod
    def from_config(cls, config: dict) -> "CointegrationAnalyzer":
        """
        Builds a CointegrationAnalyzer from the 'cointegration' block of
        config.yaml. Default construction path for pipeline code -- use
        the plain constructor directly for one-off experiments.
 
        Note: this only reads significance_level and min_observations.
        The johansen.enabled/det_order/k_ar_diff keys aren't consumed here
        since run_johansen() takes those as call-time arguments, not
        constructor state -- and johansen.enabled is currently just a
        documentation flag (no caller checks it yet, since PairSelector
        only calls run_engle_granger). Wire that check in once/if you
        build a basket-selection stage that actually calls run_johansen.
        """
        coint_cfg = config.get('cointegration', {})
        return cls(
            significance_level=coint_cfg.get('significance_level', 0.05),
            min_observations=coint_cfg.get('min_observations', 60),
        )

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Half-life of mean reversion via AR(1) regression on the spread:
            delta_spread_t = lambda * spread_{t-1} + eps_t
            half_life = -ln(2) / lambda
 
        A pair or basket can be statistically cointegrated but revert far
        too slowly to trade profitably after costs 
        """
        spread_lag = spread.shift(1).dropna()
        delta_spread = spread.diff().dropna()
        spread_lag = spread_lag.loc[delta_spread.index]

        x = sm.add_constant(spread_lag)
        model = sm.OLS(delta_spread, x).fit()
        lamb = model.params.iloc[1]

        if lamb >= 0:
            return np.inf
        return (-np.log(2)/lamb)
    
    def _test_direction(self, y: pd.Series, x: pd.Series) -> CointegrationResult:
        score, p_value, crit_values = coint(y, x, trend='c', autolag='AIC')

        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()

        intercept = model.params.iloc[0]
        hedge_ratio = model.params.iloc[1]

        spread = model.resid
        half_life = self._calculate_half_life(spread)

        return CointegrationResult(
            is_cointegrated = self.alpha > p_value,
            p_value = p_value,
            test_statistic = score,
            critical_values = {"1%": crit_values[0], "5%": crit_values[1], "10%": crit_values[2]},
            hedge_ratio = hedge_ratio,
            intercept = intercept,
            spread_series = spread,
            half_life = half_life,
            r_squared = model.rsquared,
            dependent_var = str(y.name) if y.name is not None else "y",
            independent_var = str(x.name) if x.name is not None else "x"
        )
    
    def engle_granger(self, series_y: pd.Series, series_x: pd.Series, check_both_directions: bool = True) -> CointegrationResult:
        """
        Engle-Granger 2-step test. EG is asymmetric (y-on-x != x-on-y), so
        by default this tests both directions and keeps the stronger result.
        """
        aligned_data = pd.concat([series_y, series_x], axis = 1).dropna()
        if aligned_data.shape[0] < self.min_obs:
            logger.warning(
                f"Only {aligned_data.shape[0]} obs after alignment; \n need >= {self.min_obs} for reliable testing."
            )
            return CointegrationResult(is_cointegrated=False)
        
        y, x = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]

        try: 
            result_yx = self._test_direction(y, x)
            if not check_both_directions:
                return result_yx
            
            result_xy = self._test_direction(x, y)
            winner = result_yx if result_yx.p_value <= result_xy.p_value else result_xy

            if result_xy.is_cointegrated != result_yx.is_cointegrated:
                logger.info(
                    f"EG direction sensitivity for ({y.name}, {x.name}): "
                    f"y-on-x p={result_yx.p_value:.4f}, Cointegration Status = {result_yx.is_cointegrated};"
                    f"x-on-y p={result_xy.p_value:.4f}, Cointegration Status = {result_xy.is_cointegrated}."
                )
            return winner
        except Exception as e:
            logger.error(f"Engle-Granger test failed: {e}")
            return CointegrationResult(is_cointegrated=False)

