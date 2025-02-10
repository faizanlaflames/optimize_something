# Portfolio Optimization Engine [![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

Modern portfolio optimization system implementing Markowitz's mean-variance optimization with Sharpe ratio maximization. Engineered for computational efficiency and numerical stability with O(n log n) complexity for price data processing.

## Key Features

- **Risk-Adjusted Optimization**: Maximizes Sharpe ratio using quadratic programming
- **Portfolio Analytics**: Computes 12+ performance metrics including:
  - Annualized volatility (σ)
  - Conditional Value-at-Risk (CVaR)
  - Maximum drawdown
  - Sortino ratio
- **Benchmark Comparison**: SPY ETF comparison with alpha/beta calculations
- **Monte Carlo Simulation**: Efficient frontier estimation via vectorized operations

```python
# Core optimization workflow
returns = compute_log_returns(normalized_prices)
cov_matrix = exponential_weights_covariance(returns, halflife=60)
mu = expected_returns(returns, gamma=0.5)
weights = maximize_sharpe_ratio(mu, cov_matrix, risk_free_rate=0.0)
```

## Mathematical Formulation

The optimization solves:

argmax<sub>w</sub> (w<sup>T</sup>μ - r<sub>f</sub>) / √(w<sup>T</sup>Σw)

Subject to:
- ∑w<sub>i</sub> = 1 (budget constraint)
- w<sub>i</sub> ≥ 0 (no short selling)
- σ<sub>p</sub> ≤ σ<sub>max</sub> (volatility targeting)

Where:
- μ ∈ ℝ<sup>n</sup>: Expected returns vector
- Σ ∈ ℝ<sup>n×n</sup>: Covariance matrix
- r<sub>f</sub>: Risk-free rate

## Installation

```bash
python -m venv portfolio-env
source portfolio-env/bin/activate
pip install -r requirements.txt
```

requirements.txt:
```
numpy>=1.21.0      # Vectorized operations & linear algebra
pandas>=1.3.0      # Time series processing
scipy>=1.7.0       # Convex optimization
matplotlib>=3.5.0  # Visualization
yfinance>=0.1.70   # Yahoo Finance integration
```

## Usage

### Basic Optimization
```python
from optimization import PortfolioOptimizer

optimizer = PortfolioOptimizer(
    assets=['MSFT', 'TSLA', 'BTC-USD'],
    start_date='2020-01-01',
    end_date='2023-01-01',
    risk_free_rate=0.04
)

portfolio = optimizer.optimize(volatility_target=0.2)
portfolio.summary()
```

### Advanced Configuration
```python
# Custom covariance estimation
optimizer.set_covariance_estimator(
    method='ledoit-wolf',
    prior='constant_variance'
)

# Hierarchical risk parity clustering
optimizer.configure_hrp(
    linkage_method='ward',
    covariance_type='shrunk'
)

results = optimizer.efficient_frontier(
    n_points=100,
    risk_measure='CVaR'
)
```

## Performance Metrics
Sample output for 10-asset portfolio (2020-2023):

| Metric               | Value  | Benchmark (SPY) |
|----------------------|--------|-----------------|
| Sharpe Ratio         | 1.45   | 0.89            |
| Annual Volatility    | 18.7%  | 21.3%           |
| Max Drawdown         | -24.3% | -33.8%          |
| Alpha (β=1)          | 0.09   | -               |
| Turnover Ratio       | 15.2%  | 3.1%            |

![Efficient Frontier](images/efficient_frontier.png)

## API Reference
### `PortfolioOptimizer` Class
- `optimize()`: Returns optimized weights dictionary
- `compute_risk_metrics()`: Returns ValueAtRisk, ExpectedShortfall
- `monte_carlo_simulate(n=10000)`: Returns simulated portfolio paths

### Statistical Methods
- Covariance shrinkage (Ledoit-Wolf, Oracle Approximating)
- HAC (Newey-West) covariance estimation
- Robust return estimation (Huber loss)

## References
1. Markowitz, H. (1952). Portfolio Selection
2. Ledoit, O. (2003). Honey, I Shrunk the Sample Covariance Matrix
3. Boyd, S. (2017). Convex Optimization in Portfolio Construction

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add covariance estimator'`)
4. Push branch (`git push origin feature/improvement`)
5. Open Pull Request

## License
MIT License. See [LICENSE](LICENSE) for details.

---
📫 **Quantitative Research Team** · [research@portfolio-opt.com](mailto:research@portfolio-opt.com) · [Project Wiki](https://github.com/yourrepo/wiki)
