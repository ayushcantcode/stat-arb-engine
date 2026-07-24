import pandas as pd
import numpy as np
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

# Setup module-level logger
logger = logging.getLogger(__name__)

_KPSS_SUPPORTED = {'c', 'ct'}

@dataclass
class TestResult:
    p_value: float = np.nan
    statistic: float = np.nan
    is_stationary: bool = False
    p_value_capped: bool = False # True if KPSS clipped the p-value to a table boundary
    skipped: bool = False
    error: Optional[str] = None

@dataclass
class StationarityConclusion:
    consensus_stationary: bool
    confidence: str  # "high" (unanimous), "medium" (split), "low" (insufficient data)
    score: str
    regression: str
    details: Dict[str, TestResult] = field(default_factory=dict)

class StationarityAnalyzer:
    """
    Professional-grade suite for testing time series stationarity.
    Combines ADF, KPSS, and Phillips-Perron into a joint conclusion.
 
    IMPORTANT: `regression` is applied CONSISTENTLY across all three tests.
    Running ADF/PP with 'c' (constant only) while KPSS runs 'ct' (constant
    + trend) silently asks three different questions and makes "consensus"
    meaningless. Pick the regression type that matches what you're testing:
      - 'c'  : default. Use for cointegration spread residuals (already
               mean-zero-ish by construction) or prices you don't expect
               to trend.
      - 'ct' : use for raw asset prices that plausibly have real drift.
      - 'n'  : no constant at all -- valid for ADF/PP only. KPSS will be
               SKIPPED automatically (it has no equivalent), and the
               consensus falls back to a 2-test vote.
     """

    def __init__(self, significance_level: float = 0.05, regression: str = 'c', min_obs:int = 60, strict: bool = False):
        self.alpha = significance_level
        self.regression = regression
        self.min_obs = min_obs
        # strict=True requires 3/3 agreement for consensus instead of 2/3.
        # Worth turning on if you'd rather miss tradeable pairs than admit
        # a false "stationary" -- ADF and PP share the same null hypothesis
        # and tend to move together, so a plain 2/3 majority is effectively
        # biased toward whatever ADF+PP agree on; KPSS mostly acts as a
        # tie-breaker rather than an equal third vote. strict=True forces
        # KPSS's disagreeing null hypothesis to actually carry weight.
        self.strict = strict
    
    def adf(self, series: pd.Series, regression: Optional[str] = None) -> TestResult:
        """
        Augmented Dickey-Fuller Test.
        Null Hypothesis (H0): The series has a unit root (Non-Stationary).
        """
        reg = regression or self.regression
        try:
            stat, p_value, *_ = adfuller(series.dropna(), regression=reg, autolag='AIC')
            return TestResult(p_value=p_value, statistic=stat, is_stationary=p_value < self.alpha)
        except Exception as e:
            logger.error(f"ADF test failed: {e}")
            return TestResult(error=str(e))
        
    def phillips_perron(self, series: pd.Series, regression: Optional[str] = None) -> TestResult:
        """
        Phillips-Perron Test.
        Null Hypothesis (H0): The series has a unit root (Non-Stationary).
        Robust to heteroskedasticity and serial correlation.
        """
        reg = regression or self.regression
        try:
            pp = PhillipsPerron(series.dropna(), trend=reg)
            return TestResult(p_value=pp.pvalue, statistic=pp.stat, is_stationary=pp.pvalue < self.alpha)
        except Exception as e:
            logger.error(f"Phillips-Perron test failed: {e}")
            return TestResult(error=str(e))
        
    def kpss(self, series: pd.Series, regression: Optional[str] = None) -> TestResult:
        """
        KPSS Test.
        Null Hypothesis (H0): The series is trend-stationary.
        **Notice the flipped logic compared to ADF**
        Only supports regression in {'c', 'ct'}; anything else is skipped.
        """
        reg = regression or self.regression

        if reg not in _KPSS_SUPPORTED:
            logger.info(f"KPSS doesn't support regression='{reg}' -- skipping this test.")
            return TestResult(skipped=True)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always") # Surpress Interpolation warnings
                stat, p_value, *_ = kpss(series.dropna(), regression=reg, nlags='auto')

            # KPSS p-values are interpolated from a lookup table covering
            # only [0.01, 0.1]. Outside that range the value is clipped to
            # a boundary -- catch that specific warning instead of blanket
            # suppressing everything, so we know when a p-value is exact
            # vs. just "at least this extreme".
            
            capped = any("p-value" in str(w.message).lower() for w in caught)
            return TestResult(
                p_value=p_value,
                statistic=stat,
                is_stationary=p_value >= self.alpha,
                p_value_capped=capped,
            )
        except Exception as e:
            logger.error(f"KPSS test failed {e}")
            return TestResult(error=str(e))
        

    def get_joint_conclusion(self, series: pd.Series, regression: Optional[str] = None
    ) -> StationarityConclusion:
        """
        Runs all three tests and returns a consensus dictionary.
        This is what your pipeline will actually call.
        """
        reg = regression or self.regression

        if (series.empty or (len(series.dropna()) < 30)):
            logger.warning(f"Series has {len(series.dropna())} obs; need >= {self.min_obs} for reliable testing.")
            return StationarityConclusion(
                consensus_stationary=False, confidence="low", score="0/0",
                regression=reg, details={},
            )
        
        adf_res = self.adf(series, reg)
        pp_res = self.phillips_perron(series, reg)
        kpss_res = self.kpss(series, reg)

        details = {"ADF": adf_res, "Phillips-Perron": pp_res, "KPSS": kpss_res}
 
        # Only count tests that actually ran (KPSS may have been skipped
        # for regression='n', or any test may have hit a math error).
        active = [r for r in (adf_res, pp_res, kpss_res) if not r.skipped and r.error is None]
        n_active = len(active)
 
        if n_active == 0:
            return StationarityConclusion(
                consensus_stationary=False, confidence="low", score="0/0",
                regression=reg, details=details,
            )
        votes_stationary = sum(r.is_stationary for r in active)
        required = n_active if self.strict else (n_active // 2 + 1)
        is_consensus_stationary = votes_stationary >= required
 
        if votes_stationary == n_active or votes_stationary == 0:
            confidence = "high"   # unanimous
        else:
            confidence = "medium"  # split decision
 
        return StationarityConclusion(
            consensus_stationary=is_consensus_stationary,
            confidence=confidence,
            score=f"{votes_stationary}/{n_active}",
            regression=reg,
            details=details,
        )
    
# --- Quick Test ----
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Generating a random walk (non-stationary) for the tests
    np.random.seed(42)
    random_walk = pd.Series(np.random.normal(0, 1, 1000).cumsum())
    
    # Generating white noise (Stationary)
    white_noise = pd.Series(np.random.normal(0, 1, 1000))

    trending_series = pd.Series(np.linspace(0, 50, 1000) + np.random.normal(0, 1, 1000))

    analyzer = StationarityAnalyzer()

    print("Testing Random Walk (Should be Non-Stationary):")
    print(analyzer.get_joint_conclusion(random_walk))
 
    print("\nTesting White Noise (Should be Stationary):")
    print(analyzer.get_joint_conclusion(white_noise))
 
    print("\nTesting a series with real drift, regression='c' (mismatched -- for comparison):")
    print(analyzer.get_joint_conclusion(trending_series, regression='c'))
 
    print("\nSame drifting series, regression='ct' (correct choice for a trending series):")
    print(analyzer.get_joint_conclusion(trending_series, regression='ct'))
 
    print("\nTesting with regression='n' (no constant -- correct for zero-mean spread residuals, KPSS auto-skips):")
    zero_mean_stationary = pd.Series(np.random.normal(0, 1, 1000))
    print(analyzer.get_joint_conclusion(zero_mean_stationary, regression='n'))

