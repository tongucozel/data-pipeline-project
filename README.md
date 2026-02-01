# Multi-Tier Financial Data Engineering Pipeline
### Production-Ready ETL Architecture for Quantitative Analysis

[![CI Pipeline](https://github.com/tongucozel/data-pipeline-project/actions/workflows/ci.yml/badge.svg)](https://github.com/tongucozel/data-pipeline-project/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A robust, modular, and containerized ETL (Extract, Transform, Load) pipeline designed for financial market data. This project demonstrates high-level software engineering principles, including type-safe data validation, automated CI/CD workflows, and scalable architecture.

**Live Demo:** [data-pipeline-project-tonguc.streamlit.app](https://data-pipeline-project-tonguc.streamlit.app)

## Engineering Excellence & Design Patterns

To ensure academic and professional rigor, the following principles were applied:
- **Modular Monolith Architecture**: Decoupled ETL phases allow for independent testing, maintenance, and future scalability.
- **Type-Safe Validation**: Implemented Pydantic models to enforce strict schema integrity, preventing "dirty data" from entering downstream analysis.
- **Defensive Programming**: Comprehensive logging (Standard & Debug levels) and exception handling to manage API rate limits and network inconsistencies.
- **Automated Quality Assurance**: Integrated Pytest suite with GitHub Actions to maintain high code reliability and stability.

## Tech Stack

- **Core:** Python 3.10+ (Advanced Pandas, Logging, YAML Config)
- **Data Integrity:** Pydantic (Runtime Schema Enforcement)
- **Database:** SQLite (Relational Storage with optimized indexing for time-series data)
- **Infrastructure:** Docker (Containerization for cross-platform environment parity)
- **DevOps:** GitHub Actions (CI/CD), Black/Flake8 (Linting), Mypy (Static Typing)
- **Analytics:** Plotly & Streamlit (Interactive Financial Visualizations)

## Project Structure

```
data-pipeline-project/
├── config/           # YAML-based centralized configuration management
├── src/              # Core Source Code (Modular Architecture)
│   ├── extract.py    # Robust API Integration (Yahoo Finance)
│   ├── transform.py  # Feature Engineering & Statistical Analysis
│   ├── load.py       # Database Abstraction Layer
│   └── validate.py   # Pydantic Schema Definitions for Data Quality
├── tests/            # Unit & Integration Testing Suite (Pytest)
├── data/             # Persistent Storage (Raw vs. Processed Layers)
├── main.py           # CLI & Pipeline Orchestrator Entry Point
├── Dockerfile        # Containerization Setup for Production Deployment
└── .github/          # CI/CD Workflows for Automated Testing
```

## Quick Start

### Installation & Local Setup

```bash
git clone https://github.com/tongucozel/data-pipeline-project.git
cd data-pipeline-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Execution

```bash
# Run with CLI parameters for specific symbols
python main.py --symbols NVDA META --period 1y --visualize
```

### Docker Deployment

```bash
docker build -t data-pipeline .
docker run -v $(pwd)/data:/app/data data-pipeline
```

## Analytics & Data Logic

### Data Transformation Features

The pipeline calculates key financial metrics essential for quantitative analysis:

- **Volatility Analysis**: Annualized standard deviation of daily returns.
- **Moving Averages**: 20-day (Fast) and 50-day (Slow) SMA for trend signal detection.
- **Cumulative Returns**: Performance tracking over user-defined timeframes.

### Data Validation Schema

Using Pydantic, every data point is validated against rigorous rules:

- **Type Checking**: Ensuring price data remains float and volume remains int.
- **Constraint Enforcement**: Verification of non-negative prices and non-null timestamps.

## Academic Purpose & Portfolio Note

This project was developed as a core component of my graduate school portfolio. It serves as a practical demonstration of my applied mastery in data engineering and software architecture—skills that transcend traditional undergraduate examination scores. It showcases my ability to deliver production-grade AI infrastructure, bridging the gap between theoretical knowledge and real-world implementation.

## License

MIT License - 2026.

## Author

**Tonguc Berat Ozel**
- [LinkedIn](https://www.linkedin.com/in/tongucberatozel/)
- [Live Demo](https://data-pipeline-project-tonguc.streamlit.app)
- [GitHub](https://github.com/tongucozel)
