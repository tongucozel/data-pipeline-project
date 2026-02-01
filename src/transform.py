"""
Transform Module
================
Handles data transformation and feature engineering.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transforms and enriches stock data."""

    def __init__(self, processed_data_dir: str = "data/processed"):
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def transform(
        self,
        df: pd.DataFrame,
        moving_averages: list[int] = [20, 50],
        volatility_window: int = 20,
        fill_method: str = "ffill",
        save_processed: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all transformations to the data.

        Args:
            df: Raw stock data DataFrame
            moving_averages: List of MA windows to calculate
            volatility_window: Window for volatility calculation
            fill_method: Method to fill missing values
            save_processed: Whether to save processed data

        Returns:
            Transformed DataFrame
        """
        logger.info("Starting data transformation")

        if df.empty:
            logger.warning("Empty DataFrame provided, skipping transformation")
            return df

        # Create a copy to avoid modifying original
        df = df.copy()

        # Apply transformations per symbol
        symbols = df["Symbol"].unique()
        transformed_dfs = []

        for symbol in symbols:
            symbol_df = df[df["Symbol"] == symbol].copy()
            symbol_df = self._transform_single_stock(
                symbol_df, moving_averages, volatility_window, fill_method
            )
            transformed_dfs.append(symbol_df)

        df = pd.concat(transformed_dfs, ignore_index=True)

        # Add transformation metadata
        df["TransformedAt"] = datetime.now().isoformat()

        logger.info(f"Transformation complete. Final shape: {df.shape}")

        if save_processed:
            self._save_processed_data(df)

        return df

    def _transform_single_stock(
        self,
        df: pd.DataFrame,
        moving_averages: list[int],
        volatility_window: int,
        fill_method: str,
    ) -> pd.DataFrame:
        """Transform data for a single stock."""
        symbol = df["Symbol"].iloc[0]
        logger.info(f"Transforming data for {symbol}")

        # Sort by date
        df = df.sort_values("Date").reset_index(drop=True)

        # Fill missing values
        df = self._fill_missing_values(df, fill_method)

        # Calculate moving averages
        for window in moving_averages:
            df = self._calculate_moving_average(df, window)

        # Calculate volatility
        df = self._calculate_volatility(df, volatility_window)

        # Calculate daily returns
        df = self._calculate_returns(df)

        # Calculate price change
        df = self._calculate_price_change(df)

        return df

    def _fill_missing_values(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Fill missing values in the DataFrame."""
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        existing_cols = [col for col in numeric_cols if col in df.columns]

        missing_before = df[existing_cols].isna().sum().sum()

        if method == "ffill":
            df[existing_cols] = df[existing_cols].ffill()
        elif method == "bfill":
            df[existing_cols] = df[existing_cols].bfill()
        elif method == "interpolate":
            df[existing_cols] = df[existing_cols].interpolate(method="linear")

        # Fill any remaining NaN at the start
        df[existing_cols] = df[existing_cols].bfill()

        missing_after = df[existing_cols].isna().sum().sum()
        logger.info(f"Filled {missing_before - missing_after} missing values")

        return df

    def _calculate_moving_average(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate moving average for Close price."""
        col_name = f"MA_{window}"
        df[col_name] = df["Close"].rolling(window=window).mean()
        logger.debug(f"Calculated {col_name}")
        return df

    def _calculate_volatility(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling volatility (standard deviation of returns)."""
        # Daily returns
        daily_returns = df["Close"].pct_change()
        # Rolling standard deviation (annualized)
        df["Volatility"] = daily_returns.rolling(window=window).std() * np.sqrt(252)
        logger.debug(f"Calculated Volatility with window={window}")
        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns."""
        df["Daily_Return"] = df["Close"].pct_change()
        df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
        logger.debug("Calculated Daily and Cumulative Returns")
        return df

    def _calculate_price_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price change from previous day."""
        df["Price_Change"] = df["Close"].diff()
        df["Price_Change_Pct"] = df["Close"].pct_change() * 100
        logger.debug("Calculated Price Change")
        return df

    def _save_processed_data(self, df: pd.DataFrame) -> Path:
        """Save processed data to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_data_{timestamp}.csv"
        filepath = self.processed_data_dir / filename

        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")

        return filepath
