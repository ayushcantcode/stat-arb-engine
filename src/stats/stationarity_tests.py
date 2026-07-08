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
    
    def adf(self, series: pd.series) -> dict:
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
        
    def phillips_perron(self, series: pd.series) -> dict:
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