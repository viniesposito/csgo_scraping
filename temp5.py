import pandas as pd
import numpy as np

class PortfolioBacktester:
    def __init__(self, weights_df, assets_df, costs):
        """[Previous initialization remains the same]"""
        self.initial_portfolio_value = 100  # Store this as a class variable
        # [Rest of initialization code remains unchanged]
        pass
    
    def _calculate_units(self, weights, portfolio_value, prices):
        """
        Calculate the number of units to hold based on target weights and current prices
        
        Parameters:
        weights: pd.Series of target weights (can sum to any value)
        portfolio_value: float of current portfolio value
        prices: pd.Series of current asset prices
        
        Returns:
        pd.Series of units to hold for each asset
        """
        return (weights * portfolio_value) / prices
    
    def _calculate_position_values(self, units, prices):
        """
        Calculate current position values based on units held and current prices
        
        Parameters:
        units: pd.Series of units held
        prices: pd.Series of current prices
        
        Returns:
        pd.Series of position values
        """
        return units * prices
    
    def backtest(self):
        """
        Run the backtest
        
        Returns:
        tuple of (net_backtest_df, gross_backtest_df, actual_weights_df, units_df) where:
        - net_backtest_df: DataFrame with net-of-cost portfolio values
        - gross_backtest_df: DataFrame with gross-of-cost portfolio values
        - actual_weights_df: DataFrame with actual weights over time (relative to initial portfolio value)
        - units_df: DataFrame with units held over time
        """
        # Initialize results
        portfolio_gross = pd.Series(index=self.all_dates, dtype=float)
        portfolio_net = pd.Series(index=self.all_dates, dtype=float)
        units = pd.DataFrame(index=self.all_dates, columns=self.weights_df.columns, dtype=float)
        actual_weights = pd.DataFrame(index=self.all_dates, columns=self.weights_df.columns)
        
        # Initialize portfolio values
        portfolio_gross.iloc[0] = self.initial_portfolio_value
        portfolio_net.iloc[0] = self.initial_portfolio_value
        
        # Initialize first units based on initial weights
        initial_weights = self.weights_df.iloc[0]
        current_units = self._calculate_units(
            initial_weights,
            portfolio_net.iloc[0],
            self.assets_df.iloc[0]
        )
        units.iloc[0] = current_units
        
        # Calculate initial position values and weights
        initial_position_values = self._calculate_position_values(
            current_units,
            self.assets_df.iloc[0]
        )
        # Store actual weights as position values relative to initial portfolio value
        actual_weights.iloc[0] = initial_position_values / self.initial_portfolio_value
        
        # Calculate daily servicing costs for each asset
        daily_servicing_costs = {
            asset: (1 - cost) ** (1/252) - 1 
            for asset, cost in self.costs['servicing_costs'].items()
        }
        
        # Create a map of rebalance dates to their weights for efficient lookup
        rebalance_weights = self.weights_df.to_dict('index')
        
        for i in range(1, len(self.all_dates)):
            current_date = self.all_dates[i]
            prev_date = self.all_dates[i-1]
            
            current_prices = self.assets_df.loc[current_date]
            prev_prices = self.assets_df.loc[prev_date]
            
            # Calculate current position values
            current_position_values = self._calculate_position_values(current_units, current_prices)
            portfolio_value = current_position_values.sum()
            
            # If it's a rebalancing day, update units based on new target weights
            if current_date in rebalance_weights:
                target_weights = pd.Series(rebalance_weights[current_date])
                new_units = self._calculate_units(
                    target_weights,
                    portfolio_value,
                    current_prices
                )
                
                # Calculate transaction costs based on actual traded values
                traded_values = np.abs(
                    current_position_values - 
                    self._calculate_position_values(new_units, current_prices)
                )
                transaction_costs = sum(
                    traded_values[asset] * self.costs['transaction_costs'][asset]
                    for asset in traded_values.index
                ) / portfolio_value
                
                current_units = new_units
            else:
                transaction_costs = 0
            
            # Store current units and weights
            units.loc[current_date] = current_units
            # Store weights as position values relative to initial portfolio value
            actual_weights.loc[current_date] = current_position_values / self.initial_portfolio_value
            
            # Calculate returns for each position
            asset_returns = (current_prices - prev_prices) / prev_prices
            
            # Calculate portfolio return based on position values
            prev_position_values = self._calculate_position_values(current_units, prev_prices)
            prev_portfolio_value = prev_position_values.sum()
            
            if prev_portfolio_value != 0:
                portfolio_return = (
                    (prev_position_values * asset_returns).sum() / 
                    prev_portfolio_value
                )
            else:
                portfolio_return = 0
            
            # For costs, we need relative weights within current portfolio
            position_weights = (
                current_position_values / portfolio_value 
                if portfolio_value != 0 
                else pd.Series(0, index=current_position_values.index)
            )
            daily_servicing_cost = sum(
                position_weights[asset] * daily_servicing_costs[asset]
                for asset in position_weights.index
            )
            
            # Update portfolio values
            portfolio_gross.loc[current_date] = portfolio_gross.loc[prev_date] * (1 + portfolio_return)
            portfolio_net.loc[current_date] = (
                portfolio_net.loc[prev_date] * 
                (1 + portfolio_return - daily_servicing_cost) * 
                (1 - transaction_costs)
            )
        
        # Create results DataFrames
        results_gross = pd.DataFrame({
            'value': portfolio_gross,
            'return': portfolio_gross.pct_change()
        })
        
        results_net = pd.DataFrame({
            'value': portfolio_net,
            'return': portfolio_net.pct_change()
        })
        
        return results_net, results_gross, actual_weights, units
