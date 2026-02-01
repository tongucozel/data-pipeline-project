"""
Tests for Transform Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.transform import DataTransformer


class TestDataTransformer:
    """Test cases for DataTransformer class."""

    @pytest.fixture
    def transformer(self, tmp_path):
        """Create a DataTransformer instance with temp directory."""
        return DataTransformer(processed_data_dir=str(tmp_path / "processed"))

    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100)

        # Generate realistic-looking stock data
        close_prices = 100 + np.cumsum(np.random.randn(100) * 2)

        return pd.DataFrame({
            "Date": dates,
            "Open": close_prices + np.random.randn(100),
            "High": close_prices + abs(np.random.randn(100)) + 1,
            "Low": close_prices - abs(np.random.randn(100)) - 1,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 10000000, 100),
            "Symbol": "TEST",
            "ExtractedAt": "2024-01-01T00:00:00",
        })

    def test_transformer_creates_directory(self, tmp_path):
        """Test that transformer creates processed data directory."""
        processed_dir = tmp_path / "new_processed_dir"
        transformer = DataTransformer(processed_data_dir=str(processed_dir))
        assert processed_dir.exists()

    def test_transform_adds_moving_averages(self, transformer, sample_stock_data):
        """Test that moving averages are calculated correctly."""
        df = transformer.transform(sample_stock_data, save_processed=False)

        assert "MA_20" in df.columns
        assert "MA_50" in df.columns

        # First 19 values of MA_20 should be NaN
        assert df["MA_20"].iloc[:19].isna().all()
        # Value at index 19 should not be NaN
        assert not pd.isna(df["MA_20"].iloc[19])

    def test_transform_calculates_volatility(self, transformer, sample_stock_data):
        """Test that volatility is calculated correctly."""
        df = transformer.transform(sample_stock_data, save_processed=False)

        assert "Volatility" in df.columns
        # Volatility should be positive (after initial NaN values)
        valid_volatility = df["Volatility"].dropna()
        assert (valid_volatility >= 0).all()

    def test_transform_calculates_returns(self, transformer, sample_stock_data):
        """Test that returns are calculated correctly."""
        df = transformer.transform(sample_stock_data, save_processed=False)

        assert "Daily_Return" in df.columns
        assert "Cumulative_Return" in df.columns

        # Daily return should be percentage change
        expected_return = (sample_stock_data["Close"].iloc[1] - sample_stock_data["Close"].iloc[0]) / sample_stock_data["Close"].iloc[0]
        assert abs(df["Daily_Return"].iloc[1] - expected_return) < 0.0001

    def test_transform_fills_missing_values(self, transformer, sample_stock_data):
        """Test that missing values are filled."""
        # Add some NaN values
        sample_stock_data.loc[5, "Close"] = np.nan
        sample_stock_data.loc[10, "Volume"] = np.nan

        df = transformer.transform(sample_stock_data, save_processed=False)

        # Should have no NaN in Close and Volume (except calculated columns)
        assert not df["Close"].isna().any()
        assert not df["Volume"].isna().any()

    def test_transform_adds_price_change(self, transformer, sample_stock_data):
        """Test that price change columns are added."""
        df = transformer.transform(sample_stock_data, save_processed=False)

        assert "Price_Change" in df.columns
        assert "Price_Change_Pct" in df.columns

    def test_transform_saves_processed_data(self, transformer, sample_stock_data, tmp_path):
        """Test that processed data is saved to disk."""
        transformer.transform(sample_stock_data, save_processed=True)

        processed_files = list(Path(transformer.processed_data_dir).glob("*.csv"))
        assert len(processed_files) == 1

    def test_transform_empty_dataframe(self, transformer):
        """Test that empty DataFrame is handled gracefully."""
        empty_df = pd.DataFrame()
        result = transformer.transform(empty_df, save_processed=False)

        assert result.empty

    def test_transform_adds_metadata(self, transformer, sample_stock_data):
        """Test that transformation metadata is added."""
        df = transformer.transform(sample_stock_data, save_processed=False)

        assert "TransformedAt" in df.columns

    def test_custom_moving_average_windows(self, transformer, sample_stock_data):
        """Test custom moving average windows."""
        df = transformer.transform(
            sample_stock_data,
            moving_averages=[10, 30, 100],
            save_processed=False,
        )

        assert "MA_10" in df.columns
        assert "MA_30" in df.columns
        assert "MA_100" in df.columns
        assert "MA_20" not in df.columns
        assert "MA_50" not in df.columns

    def test_multiple_symbols(self, transformer, sample_stock_data):
        """Test transformation with multiple symbols."""
        # Create data for multiple symbols
        df1 = sample_stock_data.copy()
        df1["Symbol"] = "NVDA"

        df2 = sample_stock_data.copy()
        df2["Symbol"] = "AAPL"

        combined = pd.concat([df1, df2], ignore_index=True)

        result = transformer.transform(combined, save_processed=False)

        # Should have data for both symbols
        assert set(result["Symbol"].unique()) == {"NVDA", "AAPL"}
        assert len(result) == 200  # 100 rows per symbol
