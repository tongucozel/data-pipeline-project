#!/usr/bin/env python3
"""
Data Pipeline - Main Entry Point
=================================
ETL pipeline for financial data from Yahoo Finance.

Usage:
    python main.py                          # Run with default config
    python main.py --symbols NVDA AAPL      # Specific symbols
    python main.py --period 6mo             # Different period
    python main.py --visualize              # Generate visualizations
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import yaml

from src.extract import DataExtractor
from src.transform import DataTransformer
from src.load import DataLoader
from src.validate import DataValidator, validate_database_load


def setup_logging(log_level: str = "INFO", log_file: str = "logs/pipeline.log") -> None:
    """Configure logging for the pipeline."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_visualizations(df, reports_dir: str = "reports") -> None:
    """Generate visualization reports."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("Generating visualizations...")

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    symbols = df["Symbol"].unique()

    for symbol in symbols:
        symbol_df = df[df["Symbol"] == symbol].copy()
        symbol_df["Date"] = pd.to_datetime(symbol_df["Date"])
        symbol_df = symbol_df.sort_values("Date")

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"{symbol} Stock Analysis", fontsize=14, fontweight="bold")

        # Plot 1: Price with Moving Averages
        ax1 = axes[0]
        ax1.plot(symbol_df["Date"], symbol_df["Close"], label="Close", linewidth=1.5)
        if "MA_20" in symbol_df.columns:
            ax1.plot(symbol_df["Date"], symbol_df["MA_20"], label="MA 20", linewidth=1, alpha=0.8)
        if "MA_50" in symbol_df.columns:
            ax1.plot(symbol_df["Date"], symbol_df["MA_50"], label="MA 50", linewidth=1, alpha=0.8)
        ax1.set_ylabel("Price ($)")
        ax1.legend(loc="upper left")
        ax1.set_title("Price & Moving Averages")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Plot 2: Volume
        ax2 = axes[1]
        ax2.bar(symbol_df["Date"], symbol_df["Volume"], alpha=0.7, width=1)
        ax2.set_ylabel("Volume")
        ax2.set_title("Trading Volume")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Plot 3: Volatility
        ax3 = axes[2]
        if "Volatility" in symbol_df.columns:
            ax3.plot(symbol_df["Date"], symbol_df["Volatility"], color="red", linewidth=1)
            ax3.fill_between(symbol_df["Date"], symbol_df["Volatility"], alpha=0.3, color="red")
        ax3.set_ylabel("Volatility (Annualized)")
        ax3.set_xlabel("Date")
        ax3.set_title("Rolling Volatility")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        plt.tight_layout()

        # Save figure
        output_path = reports_path / f"{symbol}_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved visualization: {output_path}")

    # Create summary statistics report
    summary_path = reports_path / "summary_statistics.csv"
    summary_stats = df.groupby("Symbol").agg({
        "Close": ["mean", "min", "max", "std"],
        "Volume": ["mean", "sum"],
        "Daily_Return": ["mean", "std"] if "Daily_Return" in df.columns else ["count"],
    }).round(2)
    summary_stats.to_csv(summary_path)
    logger.info(f"Saved summary statistics: {summary_path}")


def run_pipeline(
    symbols: list[str],
    period: str = "1y",
    interval: str = "1d",
    config: dict = None,
    visualize: bool = False,
) -> dict:
    """
    Run the complete ETL pipeline.

    Args:
        symbols: List of stock symbols to process
        period: Time period for data
        interval: Data interval
        config: Configuration dictionary
        visualize: Whether to generate visualizations

    Returns:
        Pipeline execution report
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING DATA PIPELINE")
    logger.info("=" * 60)

    start_time = datetime.now()
    report = {
        "status": "success",
        "start_time": start_time.isoformat(),
        "symbols": symbols,
        "steps": [],
    }

    try:
        # EXTRACT
        logger.info("-" * 40)
        logger.info("STEP 1: EXTRACT")
        logger.info("-" * 40)

        extractor = DataExtractor(
            raw_data_dir=config.get("output", {}).get("raw_data_dir", "data/raw")
        )
        raw_df = extractor.extract_multiple_stocks(symbols, period, interval)

        report["steps"].append({
            "step": "extract",
            "status": "success",
            "rows_extracted": len(raw_df),
        })

        if raw_df.empty:
            raise ValueError("No data extracted")

        # VALIDATE RAW DATA
        logger.info("-" * 40)
        logger.info("STEP 2: VALIDATE RAW DATA")
        logger.info("-" * 40)

        validator = DataValidator()
        validation_report = validator.validate_dataframe(raw_df)

        report["steps"].append({
            "step": "validate_raw",
            "status": "success" if validation_report["valid"] else "warning",
            "errors": validation_report["error_count"],
            "warnings": validation_report["warning_count"],
        })

        # TRANSFORM
        logger.info("-" * 40)
        logger.info("STEP 3: TRANSFORM")
        logger.info("-" * 40)

        transform_config = config.get("transform", {})
        transformer = DataTransformer(
            processed_data_dir=config.get("output", {}).get("processed_data_dir", "data/processed")
        )

        transformed_df = transformer.transform(
            raw_df,
            moving_averages=transform_config.get("moving_averages", [20, 50]),
            volatility_window=transform_config.get("volatility_window", 20),
            fill_method=transform_config.get("fill_method", "ffill"),
        )

        report["steps"].append({
            "step": "transform",
            "status": "success",
            "rows_transformed": len(transformed_df),
            "columns": list(transformed_df.columns),
        })

        # LOAD
        logger.info("-" * 40)
        logger.info("STEP 4: LOAD")
        logger.info("-" * 40)

        db_config = config.get("database", {})
        loader = DataLoader(db_path=db_config.get("path", "data/finance.db"))

        rows_loaded = loader.load_to_database(
            transformed_df,
            table_name=db_config.get("table_name", "stocks"),
        )

        report["steps"].append({
            "step": "load",
            "status": "success",
            "rows_loaded": rows_loaded,
        })

        # VALIDATE LOAD
        logger.info("-" * 40)
        logger.info("STEP 5: VALIDATE DATABASE")
        logger.info("-" * 40)

        db_validation = validate_database_load(
            db_path=db_config.get("path", "data/finance.db"),
            table_name=db_config.get("table_name", "stocks"),
            expected_symbols=symbols,
        )

        report["steps"].append({
            "step": "validate_load",
            "status": "success" if db_validation["valid"] else "failed",
            "checks": db_validation["checks"],
        })

        # VISUALIZE (optional)
        if visualize:
            logger.info("-" * 40)
            logger.info("STEP 6: VISUALIZE")
            logger.info("-" * 40)

            generate_visualizations(
                transformed_df,
                reports_dir=config.get("output", {}).get("reports_dir", "reports"),
            )

            report["steps"].append({
                "step": "visualize",
                "status": "success",
            })

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        report["status"] = "failed"
        report["error"] = str(e)
        raise

    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        report["end_time"] = end_time.isoformat()
        report["duration_seconds"] = duration

        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE - Status: {report['status']}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info("=" * 60)

    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline for Financial Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Stock symbols to fetch (e.g., NVDA AAPL GOOGL)",
    )

    parser.add_argument(
        "--period",
        type=str,
        help="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
    )

    parser.add_argument(
        "--interval",
        type=str,
        help="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization reports",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    log_config = config.get("logging", {})
    setup_logging(
        log_level=args.log_level or log_config.get("level", "INFO"),
        log_file=log_config.get("file", "logs/pipeline.log"),
    )

    # Determine symbols
    symbols = args.symbols or config.get("symbols", ["NVDA"])

    # Determine period and interval
    fetch_config = config.get("fetch", {})
    period = args.period or fetch_config.get("period", "1y")
    interval = args.interval or fetch_config.get("interval", "1d")

    # Import pandas here to avoid import error before logging setup
    global pd
    import pandas as pd

    # Run pipeline
    report = run_pipeline(
        symbols=symbols,
        period=period,
        interval=interval,
        config=config,
        visualize=args.visualize,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Status: {report['status']}")
    print(f"Duration: {report['duration_seconds']:.2f} seconds")
    print(f"Symbols: {', '.join(report['symbols'])}")
    print("\nSteps:")
    for step in report["steps"]:
        status_icon = "✓" if step["status"] == "success" else "⚠" if step["status"] == "warning" else "✗"
        print(f"  {status_icon} {step['step']}: {step['status']}")
    print("=" * 60)

    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
