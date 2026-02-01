"""
Tests for Validate Module
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

from src.validate import DataValidator, StockDataRecord, validate_database_load


class TestStockDataRecord:
    """Test cases for Pydantic model."""

    def test_valid_record(self):
        """Test valid stock data record."""
        record = StockDataRecord(
            Date="2024-01-01",
            Open=100.0,
            High=105.0,
            Low=95.0,
            Close=102.0,
            Volume=1000000,
            Symbol="NVDA",
        )
        assert record.Symbol == "NVDA"

    def test_negative_price_rejected(self):
        """Test that negative prices are rejected."""
        with pytest.raises(ValueError):
            StockDataRecord(
                Date="2024-01-01",
                Open=-100.0,  # Invalid
                High=105.0,
                Low=95.0,
                Close=102.0,
                Volume=1000000,
                Symbol="NVDA",
            )

    def test_negative_volume_rejected(self):
        """Test that negative volume is rejected."""
        with pytest.raises(ValueError):
            StockDataRecord(
                Date="2024-01-01",
                Open=100.0,
                High=105.0,
                Low=95.0,
                Close=102.0,
                Volume=-1000,  # Invalid
                Symbol="NVDA",
            )

    def test_empty_symbol_rejected(self):
        """Test that empty symbol is rejected."""
        with pytest.raises(ValueError):
            StockDataRecord(
                Date="2024-01-01",
                Open=100.0,
                High=105.0,
                Low=95.0,
                Close=102.0,
                Volume=1000000,
                Symbol="",  # Invalid
            )


class TestDataValidator:
    """Test cases for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_stock_data(self):
        """Create valid stock data for testing."""
        return pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [102.0, 103.0, 104.0],
            "Volume": [1000000, 1100000, 1200000],
            "Symbol": ["NVDA", "NVDA", "NVDA"],
        })

    def test_validate_valid_data(self, validator, valid_stock_data):
        """Test validation of valid data."""
        report = validator.validate_dataframe(valid_stock_data)

        assert report["valid"] is True
        assert report["error_count"] == 0

    def test_validate_missing_columns(self, validator):
        """Test detection of missing required columns."""
        df = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Close": [100.0],
            # Missing other required columns
        })

        report = validator.validate_dataframe(df)

        assert report["valid"] is False
        assert any(e["type"] == "missing_columns" for e in report["errors"])

    def test_validate_high_less_than_low(self, validator, valid_stock_data):
        """Test detection of High < Low."""
        valid_stock_data.loc[0, "High"] = 90.0  # Less than Low (95)
        valid_stock_data.loc[0, "Low"] = 95.0

        report = validator.validate_dataframe(valid_stock_data)

        assert report["valid"] is False
        assert any(e["type"] == "price_logic" for e in report["errors"])

    def test_validate_negative_prices(self, validator, valid_stock_data):
        """Test detection of negative prices."""
        valid_stock_data.loc[0, "Close"] = -100.0

        report = validator.validate_dataframe(valid_stock_data)

        assert report["valid"] is False
        assert any(e["type"] == "negative_price" for e in report["errors"])

    def test_validate_missing_values(self, validator, valid_stock_data):
        """Test detection of missing values in critical columns."""
        valid_stock_data.loc[0, "Close"] = np.nan

        report = validator.validate_dataframe(valid_stock_data)

        assert report["valid"] is False
        assert any(e["type"] == "missing_values" for e in report["errors"])

    def test_validate_date_gaps(self, validator, valid_stock_data):
        """Test detection of large date gaps."""
        # Create a gap larger than 5 days
        valid_stock_data.loc[2, "Date"] = "2024-01-20"

        report = validator.validate_dataframe(valid_stock_data)

        # Should produce a warning (not error)
        assert any(w["type"] == "date_gap" for w in report["warnings"])

    def test_validate_report_structure(self, validator, valid_stock_data):
        """Test that report has correct structure."""
        report = validator.validate_dataframe(valid_stock_data)

        assert "valid" in report
        assert "total_rows" in report
        assert "error_count" in report
        assert "warning_count" in report
        assert "errors" in report
        assert "warnings" in report


class TestValidateDatabaseLoad:
    """Test cases for database validation function."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a test database with sample data."""
        db_path = tmp_path / "test.db"

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE stocks (
                    Date TEXT,
                    Close REAL,
                    Symbol TEXT
                )
            """)
            conn.execute("INSERT INTO stocks VALUES ('2024-01-01', 100.0, 'NVDA')")
            conn.execute("INSERT INTO stocks VALUES ('2024-01-02', 101.0, 'NVDA')")
            conn.execute("INSERT INTO stocks VALUES ('2024-01-01', 150.0, 'AAPL')")
            conn.commit()

        return str(db_path)

    def test_validate_existing_table(self, test_db):
        """Test validation of existing table."""
        report = validate_database_load(
            db_path=test_db,
            table_name="stocks",
            expected_symbols=["NVDA", "AAPL"],
        )

        assert report["valid"] is True
        assert report["row_count"] == 3

    def test_validate_missing_table(self, test_db):
        """Test validation of non-existent table."""
        report = validate_database_load(
            db_path=test_db,
            table_name="nonexistent",
            expected_symbols=["NVDA"],
        )

        assert report["valid"] is False
        assert any(c["check"] == "table_exists" and not c["passed"] for c in report["checks"])

    def test_validate_missing_symbols(self, test_db):
        """Test validation when expected symbols are missing."""
        report = validate_database_load(
            db_path=test_db,
            table_name="stocks",
            expected_symbols=["NVDA", "AAPL", "GOOGL"],  # GOOGL is missing
        )

        assert report["valid"] is False
        assert any(c["check"] == "symbols_present" and not c["passed"] for c in report["checks"])

    def test_validate_empty_table(self, tmp_path):
        """Test validation of empty table."""
        db_path = tmp_path / "empty.db"

        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE stocks (Date TEXT, Symbol TEXT)")

        report = validate_database_load(
            db_path=str(db_path),
            table_name="stocks",
            expected_symbols=["NVDA"],
        )

        assert report["valid"] is False
        assert any(c["check"] == "has_data" and not c["passed"] for c in report["checks"])
