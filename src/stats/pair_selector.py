import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from cointegration_engine import CointegrationAnalyzer, CointegrationResult

logger = logging.getLogger(__name__)

@dataclass
class CointegratedPair:
    """A pair that has actually passed cointegration testing -- not just correlation."""
    asset_1: str
    asset_2: str
    correlation: float
    p_value: float
    hedge_ratio: float
    half_life: float

class PairSelector:
    """
    Two-stage pair selection pipeline.
 
    Stage 1 (get_correlated_pairs): a CHEAP, vectorized correlation
    pre-filter on RETURNS (not price levels), used only to shrink an
    O(N^2) universe down to a manageable candidate shortlist before the
    expensive cointegration test runs.
 
    Stage 2 (filter_by_cointegration): the REAL filter. Runs Engle-Granger
    (via CointegrationAnalyzer) on the Stage 1 shortlist and keeps only
    pairs that are genuinely cointegrated AND revert fast enough to trade.
 
    Why the split matters: correlation and cointegration are NOT the same
    thing, and conflating them is a classic pairs-trading mistake.
      - Two INDEPENDENT random walks can show high correlation purely by
        chance -- computed on price levels, two totally unrelated series
        exceed |corr| >= 0.8 about 7% of the time in simulation, just from
        both being non-stationary and trending. That false-positive rate
        drops to ~0% once correlation is computed on returns instead.
      - Two genuinely cointegrated assets (a real, stable, mean-reverting
        spread) can have LOW correlation, especially over short windows.
    Stage 1 output is a candidate list ONLY. Never trade off it directly --
    always run it through Stage 2.
    """

    def __init__(self, correlation_threshold: float = 0.6, max_half_life: float = 30.0, cointegration_analyzer: Optional[CointegrationAnalyzer] = None):
        self.corr_thresh = correlation_threshold
        self.coint_analyzer = cointegration_analyzer or CointegrationAnalyzer()
        self.max_half_life = max_half_life

    @classmethod
    def from_config(cls, config: dict) -> "PairSelector":
        """
        Builds a PairSelector from the 'pair_selection' block of
        config.yaml, and builds its CointegrationAnalyzer from the
        'cointegration' block via CointegrationAnalyzer.from_config --
        so both modules stay in sync from one config file. Default
        construction path for pipeline code; use the plain constructor
        directly for one-off experiments.
        """
        ps_cfg = config.get('pair_selection', {})
        return cls(
            correlation_threshold=ps_cfg.get('correlation_threshold', 0.6),
            max_half_life=ps_cfg.get('max_half_life_days', 30.0),
            cointegration_analyzer=CointegrationAnalyzer.from_config(config),
        )

    def get_correlated_pairs(self, price_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Vectorized correlation pre-filter computed on log returns.
        Returns a CANDIDATE list, not a verdict. Always follow with
        filter_by_cointegration() before treating any pair as tradeable.
        """
        if price_df is None or price_df.empty:
            logger.critical("Input DataFrame is empty. Upstream data pipeline failure suspected.")
            raise ValueError("Cannot compute correlation: Dataframe is empty.")
        if price_df.shape[1] < 2:
            logger.critical(f"Need at least 2 assets to find pairs. Received {price_df.shape[1]}.")
            raise ValueError("Cannot compute correlation: Insufficient assets provided.")

        returns_df = np.log(price_df / price_df.shift(1)).dropna(how='all')

        # Flag and drop zero-variance / flat-price assets before corr().
        # Otherwise they silently produce NaN correlation with everything
        # and just vanish from the output with no warning -- e.g. a halted
        # stock or a bad data pull from your DuckDB fetch would go unnoticed.

        zero_var_cols = returns_df.columns[returns_df.std(skipna=True) == 0].tolist()
        if zero_var_cols:
            logger.warning(f"Dropping zero-variance / flat-price assets: {zero_var_cols}")
            returns_df = returns_df.drop(columns = zero_var_cols)
        
        logger.info(f"Computing return - correlation matrix for {returns_df.shape[1]} assets...")

        # Compute correlation matrix and convert immediately to a NumPy array for high performance memory optimization
        corr_matrix = returns_df.corr(method='pearson').values

        # Upper triangle mask excluding the diagonal
        upper_tri_mask = np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool)
        # Creating a boolean mask of where absolute correlation meets our threshold
        threshold_mask = np.abs(corr_matrix) >= self.corr_thresh    
        # Combining the masks: We only want upper triangle AND above threshold
        valid_pairs_mask = upper_tri_mask & threshold_mask
        # Extracting the row and column indices where the mask is True
        rows, cols = np.where(valid_pairs_mask)
        # Mapping the indices back to ticker symbols directly into our final list
        tickers = returns_df.columns
        final_pairs = [
            (tickers[r], tickers[c], corr_matrix[r, c])
            for r, c in zip(rows, cols)
        ]

        final_pairs.sort(key= lambda x: abs(x[2]), reverse=True)
        logger.info(f"Found {len(final_pairs)} candidate pairs passing the >= {self.corr_thresh} threshold."
                    f"return-correlation pre-filter. CANDIDATES ONLY, run filter_by_cointegration() next.")
        return final_pairs

    def filter_by_cointegration(self, price_df: pd.DataFrame, candidate_pairs: List[Tuple[str, str, float]], max_half_life: float = 30.0) -> List[CointegratedPair]:
        """
        Runs Engle-Granger on each Stage 1 candidate (on PRICE levels --
        that part is correct, EG is specifically designed for I(1) series)
        and keeps only pairs that are:
          1. Actually cointegrated (CointegrationAnalyzer already handles
             the EG direction-asymmetry problem internally), AND
          2. Mean-reverting fast enough to be tradeable.
 
        This -- not Stage 1 correlation -- is what should gate which pairs
        move on to signal generation / backtesting.
        """

        mhl = max_half_life if max_half_life is not None else self.max_half_life
        results: List[CointegratedPair] = []

        for ticker_a, ticker_b, corr in candidate_pairs:
            eg_result: CointegrationResult = self.coint_analyzer.engle_granger(price_df[ticker_a], price_df[ticker_b])

            if not eg_result.is_cointegrated:
                continue

            if np.isnan(eg_result.half_life) or eg_result.half_life > mhl:
                logger.info(
                    f"{ticker_a}-{ticker_b}: cointegrated (p={eg_result.p_value:.4f}) "
                    f"but half-life {eg_result.half_life:.1f}d exceeds the {mhl}d cap. Therefore, skipping."
                )
                continue

            results.append(CointegratedPair(
                asset_1=ticker_a, 
                asset_2=ticker_b,
                correlation=corr,
                p_value=eg_result.p_value,
                hedge_ratio=eg_result.hedge_ratio,
                half_life=eg_result.half_life
                ))

            results.sort(key=lambda p:p.p_value)
            logger.info(
            f"{len(results)} of {len(candidate_pairs)} Stage 1 candidates passed "
            f"cointegration + half-life filtering.")
            return results

# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=250)
    asset_a = np.cumsum(np.random.normal(0, 1, 250)) + 100
    asset_b = asset_a + np.random.normal(0, 0.5, 250)
    asset_c = -asset_a + np.random.normal(0, 0.5, 250) + 200 # -ve values are invalid inside a logarithmic function (Also real prices are never -ve)
    asset_d = np.cumsum(np.random.normal(0, 2, 250)) + 50

    df = pd.DataFrame({
        'TICKER_A': asset_a,
        'TICKER_B': asset_b,
        'TICKER_C': asset_c,
        'TICKER_D': asset_d
    }, index=dates)
    
    selector = PairSelector(correlation_threshold=0.6)

    print("=== Stage 1: correlation pre-filter (candidates only) ===")
    candidates = selector.get_correlated_pairs(df)
    for p in candidates:
        print(f"{p[0]} and {p[1]} -> Corr: {p[2]:.4f}")

    print("\n=== Stage 2: real cointegration + half-life filter ===")
    confirmed = selector.filter_by_cointegration(df, candidates)
    for p in confirmed:
        print(f"  {p.asset_1}-{p.asset_2}: p={p.p_value:.4f}, hedge_ratio={p.hedge_ratio:.3f}, "
              f"half_life={p.half_life:.1f}d")
        
    # Testing Error Handling
    print("\nTesting Error Handling (Should raise ValueError):")
    try:
        selector.get_correlated_pairs(pd.DataFrame())
    except Exception as e:
        print(f"Caught expected error: {type(e).__name__} - {e}")


