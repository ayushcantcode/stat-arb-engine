import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

# Setup module-level logger
logger = logging.getLogger(__name__)

class StationarityAnalyzer:
    """
    A professional-grade suite for testing time series stationarity.
    Combines ADF, KPSS, and Phillips-Perron tests to generate a joint conclusion.
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
    
    def adf(self, series: pd.Series) -> dict:
        """
        Augmented Dickey-Fuller Test.
        Null Hypothesis (H0): The series has a unit root (Non-Stationary).
        """
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            p_value = result[1]
            is_stationary = p_value < self.alpha
            return {'p_value': p_value, 'is_stationary': is_stationary}
        except Exception as e:
            logger.error(f"ADF test failed: {e}")
            return {'p_value': np.nan, 'is_stationary': False}
        
    def phillips_perron(self, series: pd.Series) -> dict:
        """
        Phillips-Perron Test.
        Null Hypothesis (H0): The series has a unit root (Non-Stationary).
        Robust to heteroskedasticity and serial correlation.
        """
        try:
            pp = PhillipsPerron(series.dropna())
            p_value = pp.pvalue
            is_stationary = p_value < self.alpha
            return {'p_value': p_value, 'is_stationary': is_stationary} 
        except Exception as e:
            logger.error(f"Phillips-Perron test failed: {e}")
            return {'p_value': np.nan, 'is_stationary': False}
        
    def kpss(self, series: pd.Series) -> dict:
        """
        KPSS Test.
        Null Hypothesis (H0): The series is trend-stationary.
        **Notice the flipped logic compared to ADF**
        """
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Surpress Interpolation warnings
                result = kpss(series.dropna(), regression='ct', nlags='auto')
            
            p_value = result[1]
            is_stationary = p_value >= self.alpha
            return {'p_value': p_value, 'is_stationary': is_stationary}
        except Exception as e:
            logger.error(f"KPSS test failed {e}")
            return {'p_value': np.nan, 'is_stationary': False}
        

    def get_joint_conclusion(self, series: pd.Series) -> dict:
        """
        Runs all three tests and returns a consensus dictionary.
        This is what your pipeline will actually call.
        """
        if series.empty() or (len(series.dropna()) < 30):
            logger.warning("Series is too short or empty for reliable testing.")
            return {"consensus_stationary": False, "confidence": "low", "details": {}}
        
        adf_res = self.adf(series)
        kpss_res = self.kpss(series)
        pp_res = self.phillips_perron(series)

        tests_passed = sum([adf_res['is_stationary'], kpss_res['is_stationary'], pp_res['is_stationary']])
        # strict consensus: majority rules. 
        # In a highly conservative engine, you might require 3/3.
        is_consensus_stationary = tests_passed >= 2
        return {
            "consensus_stationary" : is_consensus_stationary,
            "score": f"{tests_passed}/3",
            "details": {
                "ADF": adf_res,
                "KPSS": kpss_res,
                "Phillips-Perron": pp_res
            }
        }