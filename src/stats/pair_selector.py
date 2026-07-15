import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class PairSelector:
    """
    Filters an asset universe to find highly correlated pairs 
    using vectorized NumPy operations for high-memory efficiency.
    """

    def __init__(self, correlation_threshold: float = 0.8):
        self.corr_thresh = correlation_threshold

    def get_correlated_pairs(self, price_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Calculates the correlation matrix and extracts unique pairs above the threshold.
        """
        if price_df is None or price_df.empty:
            logger.critical("Input DataFrame is empty. Upstream data pipeline failure suspected.")
            raise ValueError("Cannot compute correlation: Dataframe is empty.")
        if price_df.shape[1] < 2:
            logger.critical(f"Need at least 2 assets to find pairs. Received {price_df.shape[1]}.")
            raise ValueError("Cannot compute correlation: Insufficient assets provided.")
        
        logger.info(f"Computing correlation matrix for {price_df.shape[1]} assets...")

        # Compute correlation matrix and convert immediately to a NumPy array for high performance memory optimization
        corr_matrix = price_df.corr(method='pearson').values

        # Upper triangle mask excluding the diagonal
        upper_tri_mask = np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool)
        # Creating a boolean mask of where absolute correlation meets our threshold
        threshold_mask = np.abs(corr_matrix) >= self.corr_thresh    
        # Combining the masks: We only want upper triangle AND above threshold
        valid_pairs_mask = upper_tri_mask & threshold_mask
        # Extracting the row and column indices where the mask is True
        rows, cols = np.where(valid_pairs_mask)
        # Mapping the indices back to ticker symbols directly into our final list
        tickers = price_df.columns
        final_pairs = [
            (tickers[r], tickers[c], corr_matrix[r, c])
            for r, c in zip(rows, cols)
        ]

        final_pairs.sort(key= lambda x: abs[x[2]], reverse=True)
        logger.info(f"Found {len(final_pairs)} candidate pairs passing the >= {self.corr_threshold} threshold.")
        return final_pairs
    


