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

        final_pairs.sort(key= lambda x: abs(x[2]), reverse=True)
        logger.info(f"Found {len(final_pairs)} candidate pairs passing the >= {self.corr_thresh} threshold.")
        return final_pairs
    

# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100)
    asset_a = np.cumsum(np.random.normal(0, 1, 100))
    asset_b = asset_a + np.random.normal(0, 0.5, 100)
    asset_c = -asset_a + np.random.normal(0, 0.5, 100)
    asset_d = np.cumsum(np.random.normal(0, 2, 100))

    df = pd.DataFrame({
        'TICKER_A': asset_a,
        'TICKER_B': asset_b,
        'TICKER_C': asset_c,
        'TICKER_D': asset_d
    }, index=dates)
    
    selector = PairSelector(correlation_threshold=0.8)
    valid_pairs = selector.get_correlated_pairs(df)
    print("\n Candidate Pairs for Correlation Testing:")
    for p in valid_pairs:
        print(f"{p[0]} and {p[1]} -> Corr: {p[2]:.4f}")

    # Testing Error Handling
    print("\nTesting Error Handling (Should raise ValueError):")
    try:
        selector.get_correlated_pairs(pd.DataFrame())
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__} - {e}")


