"""
Load Module
===========
Handles loading data into SQLite database.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads data into SQLite database."""

    def __init__(self, db_path: str = "data/finance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def load_to_database(
        self,
        df: pd.DataFrame,
        table_name: str = "stocks",
        if_exists: str = "replace",
    ) -> int:
        """
        Load DataFrame into SQLite database.

        Args:
            df: DataFrame to load
            table_name: Name of the target table
            if_exists: How to behave if table exists ('fail', 'replace', 'append')

        Returns:
            Number of rows inserted
        """
        logger.info(f"Loading {len(df)} rows into table '{table_name}'")

        if df.empty:
            logger.warning("Empty DataFrame, nothing to load")
            return 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert datetime columns to string for SQLite compatibility
                df = self._prepare_for_sqlite(df)

                df.to_sql(table_name, conn, if_exists=if_exists, index=False)

                # Verify the load
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                logger.info(f"Successfully loaded data. Table '{table_name}' now has {row_count} rows")

                # Create indexes for common queries
                self._create_indexes(conn, table_name)

                return row_count

        except Exception as e:
            logger.error(f"Failed to load data into database: {str(e)}")
            raise

    def _prepare_for_sqlite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for SQLite insertion."""
        df = df.copy()

        # Convert datetime columns to string
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df

    def _create_indexes(self, conn: sqlite3.Connection, table_name: str) -> None:
        """Create indexes for better query performance."""
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(Symbol)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(Date)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_date ON {table_name}(Symbol, Date)",
        ]

        for idx_sql in indexes:
            try:
                conn.execute(idx_sql)
                logger.debug(f"Created index: {idx_sql}")
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

        conn.commit()

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.

        Args:
            sql: SQL query string

        Returns:
            Query results as DataFrame
        """
        logger.info(f"Executing query: {sql[:100]}...")

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(sql, conn)
                logger.info(f"Query returned {len(df)} rows")
                return df

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def get_table_info(self, table_name: str = "stocks") -> dict:
        """Get information about a table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get row count
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                # Get column info
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]

                # Get sample data
                sample_df = pd.read_sql_query(
                    f"SELECT * FROM {table_name} LIMIT 5", conn
                )

                return {
                    "table_name": table_name,
                    "row_count": row_count,
                    "columns": columns,
                    "sample_data": sample_df.to_dict("records"),
                }

        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            raise

    def get_latest_data(
        self, symbol: Optional[str] = None, limit: int = 10
    ) -> pd.DataFrame:
        """Get the latest stock data."""
        if symbol:
            sql = f"""
                SELECT * FROM stocks
                WHERE Symbol = '{symbol}'
                ORDER BY Date DESC
                LIMIT {limit}
            """
        else:
            sql = f"""
                SELECT * FROM stocks
                ORDER BY Date DESC
                LIMIT {limit}
            """

        return self.query(sql)
