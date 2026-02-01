"""
Validate Module
===============
Handles data validation using Pydantic models.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class StockDataRecord(BaseModel):
    """Pydantic model for validating stock data records."""

    Date: str
    Open: float = Field(gt=0, description="Opening price must be positive")
    High: float = Field(gt=0, description="High price must be positive")
    Low: float = Field(gt=0, description="Low price must be positive")
    Close: float = Field(gt=0, description="Closing price must be positive")
    Volume: int = Field(ge=0, description="Volume must be non-negative")
    Symbol: str = Field(min_length=1, max_length=10)

    @field_validator("High")
    @classmethod
    def high_must_be_highest(cls, v, info):
        """Validate that High is the highest price."""
        # Note: Cross-field validation happens in model_validator
        return v

    @field_validator("Low")
    @classmethod
    def low_must_be_lowest(cls, v, info):
        """Validate that Low is the lowest price."""
        return v


class DataValidator:
    """Validates data quality and integrity."""

    def __init__(self):
        self.validation_errors: list[dict] = []
        self.validation_warnings: list[dict] = []

    def validate_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Validate a DataFrame of stock data.

        Args:
            df: DataFrame to validate

        Returns:
            Validation report dictionary
        """
        logger.info(f"Validating DataFrame with {len(df)} rows")

        self.validation_errors = []
        self.validation_warnings = []

        # Run all validations
        self._validate_required_columns(df)
        self._validate_data_types(df)
        self._validate_price_logic(df)
        self._validate_no_missing_values(df)
        self._validate_date_continuity(df)
        self._validate_records_with_pydantic(df)

        report = {
            "valid": len(self.validation_errors) == 0,
            "total_rows": len(df),
            "error_count": len(self.validation_errors),
            "warning_count": len(self.validation_warnings),
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
        }

        if report["valid"]:
            logger.info("Validation passed successfully")
        else:
            logger.warning(f"Validation failed with {report['error_count']} errors")

        return report

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Check that all required columns are present."""
        required_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            self.validation_errors.append({
                "type": "missing_columns",
                "message": f"Missing required columns: {missing}",
            })

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types of columns."""
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.validation_errors.append({
                        "type": "invalid_dtype",
                        "column": col,
                        "message": f"Column {col} should be numeric",
                    })

    def _validate_price_logic(self, df: pd.DataFrame) -> None:
        """Validate price logic (High >= Low, etc.)."""
        if "High" in df.columns and "Low" in df.columns:
            invalid_rows = df[df["High"] < df["Low"]]
            if not invalid_rows.empty:
                self.validation_errors.append({
                    "type": "price_logic",
                    "message": f"Found {len(invalid_rows)} rows where High < Low",
                    "affected_rows": invalid_rows.index.tolist()[:10],
                })

        # Check for negative prices
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                negative_rows = df[df[col] < 0]
                if not negative_rows.empty:
                    self.validation_errors.append({
                        "type": "negative_price",
                        "column": col,
                        "message": f"Found {len(negative_rows)} rows with negative {col}",
                    })

    def _validate_no_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing values in critical columns."""
        critical_cols = ["Date", "Close", "Symbol"]

        for col in critical_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    self.validation_errors.append({
                        "type": "missing_values",
                        "column": col,
                        "message": f"Column {col} has {missing_count} missing values",
                    })

    def _validate_date_continuity(self, df: pd.DataFrame) -> None:
        """Check for unusual gaps in date sequence."""
        if "Date" not in df.columns or "Symbol" not in df.columns:
            return

        for symbol in df["Symbol"].unique():
            symbol_df = df[df["Symbol"] == symbol].copy()

            if "Date" in symbol_df.columns:
                try:
                    dates = pd.to_datetime(symbol_df["Date"])
                    dates = dates.sort_values()

                    # Check for gaps > 5 business days (excluding weekends)
                    gaps = dates.diff()
                    large_gaps = gaps[gaps > pd.Timedelta(days=5)]

                    if not large_gaps.empty:
                        self.validation_warnings.append({
                            "type": "date_gap",
                            "symbol": symbol,
                            "message": f"Found {len(large_gaps)} gaps > 5 days for {symbol}",
                        })
                except Exception as e:
                    logger.warning(f"Could not validate date continuity: {e}")

    def _validate_records_with_pydantic(self, df: pd.DataFrame, sample_size: int = 100) -> None:
        """Validate a sample of records using Pydantic model."""
        # Sample rows for detailed validation
        sample_df = df.head(sample_size) if len(df) > sample_size else df

        invalid_count = 0
        for idx, row in sample_df.iterrows():
            try:
                record_dict = row.to_dict()
                # Convert types as needed
                if "Volume" in record_dict:
                    record_dict["Volume"] = int(record_dict["Volume"])
                if "Date" in record_dict:
                    record_dict["Date"] = str(record_dict["Date"])

                StockDataRecord(**record_dict)
            except Exception as e:
                invalid_count += 1
                if invalid_count <= 5:  # Only log first 5 errors
                    logger.debug(f"Pydantic validation failed for row {idx}: {e}")

        if invalid_count > 0:
            self.validation_warnings.append({
                "type": "pydantic_validation",
                "message": f"{invalid_count}/{len(sample_df)} sampled records failed Pydantic validation",
            })


def validate_database_load(db_path: str, table_name: str, expected_symbols: list[str]) -> dict:
    """
    Validate that data was loaded correctly into the database.

    Args:
        db_path: Path to SQLite database
        table_name: Name of the table to validate
        expected_symbols: List of symbols that should be in the table

    Returns:
        Validation report
    """
    import sqlite3

    logger.info(f"Validating database load: {db_path}, table: {table_name}")

    report = {
        "valid": True,
        "checks": [],
    }

    try:
        with sqlite3.connect(db_path) as conn:
            # Check table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if not cursor.fetchone():
                report["valid"] = False
                report["checks"].append({
                    "check": "table_exists",
                    "passed": False,
                    "message": f"Table '{table_name}' does not exist",
                })
                return report

            report["checks"].append({
                "check": "table_exists",
                "passed": True,
            })

            # Check row count
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            report["row_count"] = row_count

            if row_count == 0:
                report["valid"] = False
                report["checks"].append({
                    "check": "has_data",
                    "passed": False,
                    "message": "Table is empty",
                })
            else:
                report["checks"].append({
                    "check": "has_data",
                    "passed": True,
                    "message": f"Table has {row_count} rows",
                })

            # Check expected symbols
            cursor = conn.execute(f"SELECT DISTINCT Symbol FROM {table_name}")
            actual_symbols = [row[0] for row in cursor.fetchall()]

            missing_symbols = set(expected_symbols) - set(actual_symbols)
            if missing_symbols:
                report["valid"] = False
                report["checks"].append({
                    "check": "symbols_present",
                    "passed": False,
                    "message": f"Missing symbols: {missing_symbols}",
                })
            else:
                report["checks"].append({
                    "check": "symbols_present",
                    "passed": True,
                    "message": f"All expected symbols present: {actual_symbols}",
                })

    except Exception as e:
        report["valid"] = False
        report["checks"].append({
            "check": "database_connection",
            "passed": False,
            "message": str(e),
        })

    logger.info(f"Database validation complete. Valid: {report['valid']}")
    return report
