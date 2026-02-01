"""
Extract Module
==============
Handles data extraction from Yahoo Finance API.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extracts stock data from Yahoo Finance."""

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def extract_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        save_raw: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Extract stock data for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'NVDA')
            period: Data period (e.g., '1y', '6mo', '1mo')
            interval: Data interval (e.g., '1d', '1h')
            save_raw: Whether to save raw data to disk

        Returns:
            DataFrame with stock data or None if extraction fails
        """
        logger.info(f"Extracting data for {symbol} (period={period}, interval={interval})")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Add metadata columns
            df["Symbol"] = symbol
            df["ExtractedAt"] = datetime.now().isoformat()

            # Reset index to make Date a column
            df = df.reset_index()

            logger.info(f"Successfully extracted {len(df)} rows for {symbol}")

            if save_raw:
                self._save_raw_data(df, symbol)

            return df

        except Exception as e:
            logger.error(f"Failed to extract data for {symbol}: {str(e)}")
            raise

    def _save_raw_data(self, df: pd.DataFrame, symbol: str) -> Path:
        """Save raw data to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_raw_{timestamp}.csv"
        filepath = self.raw_data_dir / filename

        df.to_csv(filepath, index=False)
        logger.info(f"Raw data saved to {filepath}")

        return filepath

    def extract_multiple_stocks(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Extract data for multiple stock symbols.

        Args:
            symbols: List of stock ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            Combined DataFrame with all stock data
        """
        logger.info(f"Extracting data for {len(symbols)} symbols: {symbols}")

        all_data = []
        for symbol in symbols:
            try:
                df = self.extract_stock_data(symbol, period, interval)
                if df is not None:
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Skipping {symbol} due to error: {e}")
                continue

        if not all_data:
            logger.warning("No data extracted for any symbol")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data: {len(combined_df)} total rows")

        return combined_df
