""""""  		  	   		 	 	 			  		 			     			  	 
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Faizan Hussain 		  	   		 	 	 			  		 			     			  	 
GT User ID: fhussain45	  	   		 	 	 			  		 			     			  	 
GT ID: 904082279 		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	  	   		 	 	 			  		 			     			  	 
import matplotlib.pyplot as plt  	  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  
from scipy.optimize import minimize		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			
                    

def author():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return "fhussain45"  		  	   

def study_group():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: comma seperated string of each gtname in my study group  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return "fhussain45"   		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
# This is the function that will be tested by the autograder  		  	   		 	 	 			  		 			     			  	 
# The student must update this code to properly implement the functionality  		  	   		 	 	 			  		 			     			  	 
def optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    ed=dt.datetime(2009, 1, 1),  		  	   		 	 	 			  		 			     			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	 	 			  		 			     			  	 
    gen_plot=True,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 			  		 			     			  	 
    statistics.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 			  		 			     			  	 
    :type sd: datetime  		  	   		 	 	 			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	 	 			  		 			     			  	 
    :type ed: datetime  		  	   		 	 	 			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 			  		 			     			  	 
        symbol in the data directory)  		  	   		 	 	 			  		 			     			  	 
    :type syms: list  		  	   		 	 	 			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 			  		 			     			  	 
        code with gen_plot = False.  		  	   		 	 	 			  		 			     			  	 
    :type gen_plot: bool  		  	   		 	 	 			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 			  		 			     			  	 
    :rtype: tuple  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	 	 			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		 	 	 			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	 	 			  		 			     			  	 
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	 	 			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # find the allocations for the optimal portfolio  		  	   		 	 	 			  		 			     			  	  		  	   		 	 	 			  		 			     			  	 
   
    # normalizing prices
    normalized_prices = prices / prices.iloc[0]

    # mostly for port. optimization
    def negative_sharpe(allocs):
        port_val = (normalized_prices * allocs).sum(axis=1)
        port_daily_returns = port_val.pct_change().dropna()
        mean_return = port_daily_returns.mean()
        std_return = port_daily_returns.std()
        sharpe_ratio = np.sqrt(252) * mean_return / std_return
        return -sharpe_ratio  
    
    # constraints, allocations must sum to 1 
    constraints = ({'type': 'eq', 'fun': lambda allocs: np.sum(allocs) - 1})
    
    # defining our bounds
    bounds = [(0.0, 1.0) for _ in syms]
    
    # first guess, equal allocation
    first_guess = np.ones(len(syms)) / len(syms)
    

    #running the optimization w/minimize from scipy
    result = minimize(negative_sharpe, first_guess, method='SLSQP', bounds = bounds, constraints=constraints)
    optimized_allocs = result.x 
    
    # calculating statistics
    port_val = (normalized_prices * optimized_allocs).sum(axis=1)
    port_daily_returns = port_val.pct_change().dropna()
    cr = (port_val[-1] / port_val[0]) - 1
    adr = port_daily_returns.mean()
    sddr = port_daily_returns.std()
    sr =  adr / sddr
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	 	 			  		 			     			  	 
    if gen_plot:  		  	   		 	 	 			  		 			     			  	 
        normalized_SPY = prices_all['SPY'] / prices_all['SPY'].iloc[0] 		  	   		 	 	 			  		 			     			  	 
        df_temp = pd.concat(  		  	   		 	 	 			  		 			     			  	 
            [port_val / port_val.iloc[0], normalized_SPY], axis=1, keys=['Portfolio', 'SPY']  		  	   		 	 	 			  		 			     			  	 
        )  	
        df_temp.plot(title="Daily Portfolio Value and SPY", xlabel="Date", ylabel="Price")	  	   		 	 	 			  		 			     			  	 
        plt.legend()
        plt.savefig("Figure1.png") #save as png
        # plt.show()
        plt.close()		   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    return optimized_allocs, cr, adr, sddr, sr  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  	#to generate report plot	  	   		 	 	 			  		 			     			  	 
    start_date = dt.datetime(2008, 6, 1)  		  	   		 	 	 			  		 			     			  	 
    end_date = dt.datetime(2009, 6, 1)  		  	   		 	 	 			  		 			     			  	 
    symbols = ["IBM", "X", "GLD", "JPM",]  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Assess the portfolio  		  	   		 	 	 			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	 	 			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True  		  	   		 	 	 			  		 			     			  	 
    )  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Print statistics  		  	   		 	 	 			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		 	 	 			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return: {cr}")   		 	 	 			  		 			     			  	 		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 			  		 			     			  	 
    # Do not assume that it will be called  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
