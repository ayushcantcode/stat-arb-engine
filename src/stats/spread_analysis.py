import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

class SpreadAnalyzer:
    """
    Signal generation engine for a single cointegrated pair.
 
    Pipeline: calculate_spread -> calculate_zscore -> generate_signals,
    or call generate() to run all three in one pass.
 
    LOOK-AHEAD BIAS WARNING: hedge_ratio must not be informed by data from
    the period you're generating signals for. Two valid patterns:
      (a) Fit hedge_ratio ONCE via CointegrationAnalyzer on a formation
          window, then call generate_signals only on the SUBSEQUENT
          trading window (a static float).
      (b) Use calculate_rolling_hedge_ratio() so the ratio at time t only
          ever uses data up to and including t (a pd.Series).
    Passing a hedge ratio fit on the same window you're testing signals
    against will produce backtest results that look better than anything
    achievable in live trading.
    """

    def __init__(self, lookback_window: int = 30, entry_z: float = 2.0, exit_z: float = 0.0, stop_loss_z: Optional[float] = 4.0):
        """
        Args:
            lookback_window: rolling window for the z-score's mean/std.
            entry_z: |z| threshold to open a position.
            exit_z: z threshold (crossing back toward it) to close a
                position on mean reversion. Usually 0.0.
            stop_loss_z: |z| threshold beyond which a position is force-
                closed regardless of the mean-reversion logic. Cointegration
                can break down (see CointegrationAnalyzer / PairSelector) --
                without this, a position that stops reverting because the
                underlying relationship broke will sit accumulating
                unbounded loss while waiting for a reversion that may never
                come. Pass None to explicitly disable (not recommended).
        """
        if lookback_window < 2:
            raise ValueError("lookback_window must be >= 2")
        if entry_z <= exit_z:
            raise ValueError(f"entry_z ({entry_z}) must be strictly greater than exit_z ({exit_z}). "
                "Otherwise a freshly entered position would immediately satisfy its own exit condition on the very next check.")
        if stop_loss_z is not None and stop_loss_z <= entry_z:
            raise ValueError(
                f"stop_loss_z ({stop_loss_z}) must be greater than entry_z ({entry_z}), "
                "otherwise every entry would trip the stop-loss immediately.")

        self.window = lookback_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z

    @classmethod
    def from_config(cls, config: dict) -> "SpreadAnalyzer":
        """
        Builds a SpreadAnalyzer from the 'signal_generation' block of
        config.yaml. Default construction path for pipeline code -> use
        the plain constructor directly for one-off experiments.
 
        Note: stop_loss_z can be set to `null` in the yaml to explicitly
        disable it (config.get returns None for an explicit null, same as
        passing stop_loss_z=None directly), distinct from the key being
        absent entirely, which falls back to the 4.0 default below.
        """
        cfg = config.get('signal_generation', {})
        return cls(
            lookback_window = cfg.get("lookback_window", 30),
            entry_z = cfg.get("entry_z", 2.0),
            exit_z = cfg.get("exit_z", 0.0),
            stop_loss_z = cfg.get("stop_loss_z", 4.0),
        )

def calculate_spread(self, prices: pd.DataFrame, pair: Tuple[str, str], hedge_ratio: Union[float, pd.Series]) -> pd.Series:
    """
        Constructs the spread: Spread = Asset_Y - (Hedge_Ratio * Asset_X).
 
        hedge_ratio can be a static float (formation-period fit) or a
        pd.Series aligned to prices.index (from
        calculate_rolling_hedge_ratio()) for a time-varying hedge.
        """

    if len(pair) != 2:
        raise ValueError(f"pair must contain exactly 2 tickers, got {pair}.")
    asset_y, asset_x = pair

    missing = [t for t in pair if t not in prices.columns]
    if missing:
        raise KeyError(f"Ticker(s) {missing} not found in prices.columns.")

    if prices[[asset_y, asset_x]].isna().any().any():
        n_nan = prices[[asset_y, asset_x]].isna().any(axis=1).sum()
        logger.warning(
                f"{n_nan} rows have NaN price data for {pair}, spread will be "
                f"NaN at those dates rather than silently misaligning."
            )        

    if isinstance(hedge_ratio, pd.Series):
        hedge_ratio = hedge_ratio.reindex(prices.index)
        if hedge_ratio.isna().any():
            logger.warning(
                    f"Rolling hedge_ratio has {hedge_ratio.isna().sum()} NaNs after "
                    f"aligning to prices.index (expected during its own warmup period) "
                    f"spread will be NaN there too."
                )
    else:
        if not np.isfinite(hedge_ratio):
            raise ValueError(f"hedge_ratio must be finite, got {hedge_ratio}.")

    spread = prices[asset_y] - (hedge_ratio * prices[asset_x])
    spread.name = f"spread_{asset_y}_{asset_x}"
    return spread