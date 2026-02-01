"""
Tests for Extract Module
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from src.extract import DataExtractor


class TestDataExtractor:
    """Test cases for DataExtractor class."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a DataExtractor instance with temp directory."""
        return DataExtractor(raw_data_dir=str(tmp_path / "raw"))

    @pytest.fixture
    def mock_stock_data(self):
        """Create mock stock data."""
        return pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low": [95.0] * 10,
            "Close": [102.0] * 10,
            "Volume": [1000000] * 10,
        }).set_index("Date")

    def test_extractor_creates_directory(self, tmp_path):
        """Test that extractor creates raw data directory."""
        raw_dir = tmp_path / "new_raw_dir"
        extractor = DataExtractor(raw_data_dir=str(raw_dir))
        assert raw_dir.exists()

    @patch("src.extract.yf.Ticker")
    def test_extract_stock_data_success(self, mock_ticker, extractor, mock_stock_data):
        """Test successful stock data extraction."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_stock_data
        mock_ticker.return_value = mock_ticker_instance

        df = extractor.extract_stock_data("NVDA", save_raw=False)

        assert df is not None
        assert len(df) == 10
        assert "Symbol" in df.columns
        assert "ExtractedAt" in df.columns
        assert df["Symbol"].iloc[0] == "NVDA"

    @patch("src.extract.yf.Ticker")
    def test_extract_stock_data_empty(self, mock_ticker, extractor):
        """Test extraction when no data is returned."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        df = extractor.extract_stock_data("INVALID", save_raw=False)

        assert df is None

    @patch("src.extract.yf.Ticker")
    def test_extract_saves_raw_data(self, mock_ticker, extractor, mock_stock_data, tmp_path):
        """Test that raw data is saved to disk."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_stock_data
        mock_ticker.return_value = mock_ticker_instance

        extractor.extract_stock_data("NVDA", save_raw=True)

        raw_files = list(Path(extractor.raw_data_dir).glob("*.csv"))
        assert len(raw_files) == 1
        assert "NVDA" in raw_files[0].name

    @patch("src.extract.yf.Ticker")
    def test_extract_multiple_stocks(self, mock_ticker, extractor, mock_stock_data):
        """Test extracting data for multiple stocks."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_stock_data
        mock_ticker.return_value = mock_ticker_instance

        df = extractor.extract_multiple_stocks(["NVDA", "AAPL"])

        # Should have data for both symbols (though mocked, they'll have same data)
        assert len(df) == 20  # 10 rows * 2 symbols

    @patch("src.extract.yf.Ticker")
    def test_extract_handles_api_error(self, mock_ticker, extractor):
        """Test that API errors are handled properly."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_ticker_instance

        with pytest.raises(Exception, match="API Error"):
            extractor.extract_stock_data("NVDA", save_raw=False)


class TestExtractIntegration:
    """Integration tests for extract module (requires network)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_api_call(self, tmp_path):
        """Test actual API call to Yahoo Finance."""
        extractor = DataExtractor(raw_data_dir=str(tmp_path / "raw"))
        df = extractor.extract_stock_data("AAPL", period="5d", save_raw=False)

        assert df is not None
        assert len(df) > 0
        assert "Close" in df.columns
