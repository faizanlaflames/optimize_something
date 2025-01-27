# Portfolio Optimization

This project implements a portfolio optimization algorithm using the Sharpe ratio as the objective function. It finds the optimal allocation of assets to maximize the risk-adjusted return of a portfolio.

## Features

- Optimizes portfolio allocations for a given set of stocks
- Maximizes Sharpe ratio using scipy's minimize function
- Generates performance metrics including cumulative return, average daily return, and volatility
- Optional plot generation comparing portfolio performance to SPY benchmark

## Installation

```bash
pip install numpy pandas matplotlib scipy
```

## Usage

```python
from optimization import optimize_portfolio
import datetime as dt

start_date = dt.datetime(2008, 6, 1)
end_date = dt.datetime(2009, 6, 1)
symbols = ["IBM", "X", "GLD", "JPM"]

allocations, cr, adr, sddr, sr = optimize_portfolio(
    sd=start_date, 
    ed=end_date, 
    syms=symbols, 
    gen_plot=True
)

print(f"Allocations: {allocations}")
print(f"Sharpe Ratio: {sr}")
print(f"Volatility: {sddr}")
print(f"Average Daily Return: {adr}")
print(f"Cumulative Return: {cr}")
```

## Implementation Details

- Uses `get_data` function to fetch adjusted closing prices
- Normalizes prices and calculates daily returns
- Defines negative Sharpe ratio as the objective function for minimization
- Applies constraints to ensure allocations sum to 1
- Utilizes scipy's `minimize` function with SLSQP method for optimization
- Calculates portfolio statistics based on optimized allocations

## Dependencies

- numpy
- pandas
- matplotlib
- scipy
