"""
Tests for Load Module
"""

import pytest
import pandas as pd
import sqlite3
from pathlib import Path

from src.load import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create a DataLoader instance with temp database."""
        db_path = tmp_path / "test_finance.db"
        return DataLoader(db_path=str(db_path))

    @pytest.fixture
    def sample_transformed_data(self):
        """Create sample transformed data for testing."""
        return pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [102.0, 103.0, 104.0],
            "Volume": [1000000, 1100000, 1200000],
            "Symbol": ["NVDA", "NVDA", "NVDA"],
            "MA_20": [100.0, 100.5, 101.0],
            "MA_50": [99.0, 99.5, 100.0],
            "Volatility": [0.2, 0.21, 0.22],
            "Daily_Return": [0.0, 0.01, 0.01],
            "TransformedAt": ["2024-01-01T12:00:00"] * 3,
        })

    def test_loader_creates_directory(self, tmp_path):
        """Test that loader creates database directory."""
        db_path = tmp_path / "subdir" / "finance.db"
        loader = DataLoader(db_path=str(db_path))
        assert db_path.parent.exists()

    def test_load_to_database(self, loader, sample_transformed_data):
        """Test loading data into database."""
        rows = loader.load_to_database(sample_transformed_data, table_name="stocks")

        assert rows == 3

        # Verify data was loaded
        with sqlite3.connect(loader.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM stocks", conn)
            assert len(df) == 3
            assert "Close" in df.columns

    def test_load_creates_indexes(self, loader, sample_transformed_data):
        """Test that indexes are created."""
        loader.load_to_database(sample_transformed_data, table_name="stocks")

        with sqlite3.connect(loader.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='stocks'"
            )
            indexes = [row[0] for row in cursor.fetchall()]

            assert "idx_stocks_symbol" in indexes
            assert "idx_stocks_date" in indexes

    def test_load_empty_dataframe(self, loader):
        """Test loading empty DataFrame."""
        empty_df = pd.DataFrame()
        rows = loader.load_to_database(empty_df, table_name="stocks")

        assert rows == 0

    def test_load_replace_existing(self, loader, sample_transformed_data):
        """Test replacing existing data."""
        # First load
        loader.load_to_database(sample_transformed_data, table_name="stocks")

        # Second load with replace
        new_data = sample_transformed_data.copy()
        new_data["Close"] = [200.0, 201.0, 202.0]
        loader.load_to_database(new_data, table_name="stocks", if_exists="replace")

        # Should have replaced data
        with sqlite3.connect(loader.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM stocks", conn)
            assert df["Close"].iloc[0] == 200.0

    def test_load_append_existing(self, loader, sample_transformed_data):
        """Test appending to existing data."""
        # First load
        loader.load_to_database(sample_transformed_data, table_name="stocks")

        # Second load with append
        loader.load_to_database(sample_transformed_data, table_name="stocks", if_exists="append")

        # Should have doubled the data
        with sqlite3.connect(loader.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM stocks", conn)
            assert len(df) == 6

    def test_query(self, loader, sample_transformed_data):
        """Test querying the database."""
        loader.load_to_database(sample_transformed_data, table_name="stocks")

        df = loader.query("SELECT * FROM stocks WHERE Close > 102")

        assert len(df) == 2

    def test_get_table_info(self, loader, sample_transformed_data):
        """Test getting table information."""
        loader.load_to_database(sample_transformed_data, table_name="stocks")

        info = loader.get_table_info("stocks")

        assert info["table_name"] == "stocks"
        assert info["row_count"] == 3
        assert len(info["columns"]) > 0
        assert len(info["sample_data"]) <= 5

    def test_get_latest_data(self, loader, sample_transformed_data):
        """Test getting latest data."""
        loader.load_to_database(sample_transformed_data, table_name="stocks")

        df = loader.get_latest_data(limit=2)
        assert len(df) == 2

    def test_get_latest_data_by_symbol(self, loader, sample_transformed_data):
        """Test getting latest data for specific symbol."""
        # Add another symbol
        multi_symbol = sample_transformed_data.copy()
        aapl_data = sample_transformed_data.copy()
        aapl_data["Symbol"] = "AAPL"
        combined = pd.concat([multi_symbol, aapl_data], ignore_index=True)

        loader.load_to_database(combined, table_name="stocks")

        df = loader.get_latest_data(symbol="NVDA", limit=10)
        assert all(df["Symbol"] == "NVDA")


class TestLoadIntegration:
    """Integration tests for load module."""

    def test_full_load_and_query_workflow(self, tmp_path):
        """Test complete load and query workflow."""
        db_path = tmp_path / "integration_test.db"
        loader = DataLoader(db_path=str(db_path))

        # Create test data
        data = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=100).astype(str),
            "Open": range(100),
            "High": range(1, 101),
            "Low": range(100),
            "Close": range(100),
            "Volume": [1000000] * 100,
            "Symbol": ["TEST"] * 100,
        })

        # Load
        loader.load_to_database(data, table_name="stocks")

        # Query
        result = loader.query("SELECT AVG(Close) as avg_close FROM stocks")
        assert result["avg_close"].iloc[0] == 49.5  # Average of 0-99

        # Get info
        info = loader.get_table_info("stocks")
        assert info["row_count"] == 100
