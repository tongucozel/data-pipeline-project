# Financial Data ETL Pipeline

[![CI Pipeline](https://github.com/YOUR_USERNAME/data-pipeline-project/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/data-pipeline-project/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready ETL (Extract, Transform, Load) pipeline for financial stock data, built with Python. Fetches data from Yahoo Finance, performs feature engineering, and stores results in SQLite.

## Features

- **Extract**: Fetch historical stock data from Yahoo Finance API
- **Transform**: Calculate moving averages, volatility, returns, and more
- **Load**: Store processed data in SQLite with proper indexing
- **Validate**: Data quality checks with Pydantic models
- **Visualize**: Generate analysis charts and reports
- **CLI**: Flexible command-line interface
- **Docker**: Containerized for easy deployment
- **CI/CD**: GitHub Actions for automated testing

## Project Structure

```
data-pipeline-project/
├── config/
│   └── config.yaml          # Pipeline configuration
├── src/
│   ├── __init__.py
│   ├── extract.py            # Data extraction from Yahoo Finance
│   ├── transform.py          # Data transformation & feature engineering
│   ├── load.py               # Database operations
│   └── validate.py           # Data validation with Pydantic
├── tests/
│   ├── test_extract.py
│   ├── test_transform.py
│   ├── test_load.py
│   └── test_validate.py
├── data/
│   ├── raw/                  # Raw extracted data
│   └── processed/            # Transformed data
├── logs/                     # Pipeline logs
├── reports/                  # Generated visualizations
├── main.py                   # CLI entry point
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/data-pipeline-project.git
cd data-pipeline-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default configuration (NVDA, AAPL, GOOGL - 1 year)
python main.py

# Run for specific symbols
python main.py --symbols NVDA TSLA MSFT

# Run with different time period
python main.py --symbols NVDA --period 6mo

# Generate visualizations
python main.py --symbols NVDA AAPL --visualize

# Debug mode
python main.py --log-level DEBUG
```

### Docker Usage

```bash
# Build the image
docker build -t data-pipeline .

# Run the container
docker run -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports data-pipeline

# Run with custom symbols
docker run data-pipeline python main.py --symbols MSFT --period 3mo
```

## Configuration

Edit `config/config.yaml` to customize the pipeline:

```yaml
symbols:
  - NVDA
  - AAPL
  - GOOGL

fetch:
  period: "1y"
  interval: "1d"

transform:
  moving_averages: [20, 50]
  volatility_window: 20

database:
  path: "data/finance.db"
  table_name: "stocks"
```

## Output

### Database Schema

The pipeline creates a `stocks` table with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Date | TEXT | Trading date |
| Open | REAL | Opening price |
| High | REAL | Highest price |
| Low | REAL | Lowest price |
| Close | REAL | Closing price |
| Volume | INTEGER | Trading volume |
| Symbol | TEXT | Stock ticker |
| MA_20 | REAL | 20-day moving average |
| MA_50 | REAL | 50-day moving average |
| Volatility | REAL | Annualized volatility |
| Daily_Return | REAL | Daily return |
| Cumulative_Return | REAL | Cumulative return |

### Visualizations

When run with `--visualize`, the pipeline generates:
- Price charts with moving averages
- Volume charts
- Volatility analysis
- Summary statistics CSV

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_transform.py -v

# Run integration tests (requires network)
pytest tests/ -v -m integration
```

## Development

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Technical Decisions

1. **SQLite over PostgreSQL**: Chosen for portability and zero-config setup
2. **Pydantic for Validation**: Type-safe data validation with clear error messages
3. **YAML Configuration**: Human-readable and easy to modify
4. **Modular Architecture**: Each ETL step is independent and testable
5. **Comprehensive Logging**: Full audit trail for debugging

## License

MIT License - see LICENSE file for details.

## Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
