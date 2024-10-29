import pandas as pd
import numpy as np

class PortfolioBacktester:
    def __init__(self, weights_df, assets_df, costs):
        """
        Initialize the backtester with input data
        
        Parameters:
        weights_df: pd.DataFrame with dates as index and assets as columns, containing desired weights
        assets_df: pd.DataFrame with dates as index and assets as columns, containing asset prices
        costs: dict with keys:
            - 'transaction_costs': dict mapping asset names to their transaction costs (as fractions)
            - 'servicing_costs': dict mapping asset names to their annual servicing costs (as fractions)
        """
        # Sort columns alphabetically in both DataFrames to ensure consistent ordering
        self.weights_df = weights_df.copy().sort_index(axis=1)
        self.assets_df = assets_df.copy().sort_index(axis=1)
        
        # Validate that all assets in weights have corresponding price data
        missing_assets = set(self.weights_df.columns) - set(self.assets_df.columns)
        if missing_assets:
            raise ValueError(f"Missing price data for assets: {missing_assets}")
        
        # Remove any excess columns from assets_df that aren't in weights_df
        self.assets_df = self.assets_df[self.weights_df.columns]
        
        # Determine the backtest period
        self._determine_backtest_period()
        
        # Validate costs structure
        required_assets = set(self.weights_df.columns)
        for cost_type in ['transaction_costs', 'servicing_costs']:
            if cost_type not in costs:
                raise ValueError(f"Missing {cost_type} in costs dictionary")
            missing_costs = required_assets - set(costs[cost_type].keys())
            if missing_costs:
                raise ValueError(f"Missing {cost_type} for assets: {missing_costs}")
        
        self.costs = costs
            
        # Print initial setup information
        print("\nBacktest Configuration:")
        print(f"Full asset data range: {self.assets_df.index[0]} to {self.assets_df.index[-1]}")
        print(f"Original weight dates: {self.weights_df.index[0]} to {self.weights_df.index[-1]}")
        print(f"Actual backtest period: {self.backtest_start_date} to {self.backtest_end_date}")
        print(f"Number of assets: {len(self.weights_df.columns)}")
        print(f"Assets being used (in order): {', '.join(self.weights_df.columns)}\n")
    
    def _determine_backtest_period(self):
        """
        Determine the valid backtest period based on available data
        """
        # Find the first rebalance date
        first_weight_date = self.weights_df.index[0]
        
        # Find the first available trading date on or after the first weight date
        valid_dates = self.assets_df.index[self.assets_df.index >= first_weight_date]
        if len(valid_dates) == 0:
            raise ValueError("No asset data available after first weight date")
        
        self.backtest_start_date = valid_dates[0]
        
        # Find the last rebalance date
        last_weight_date = self.weights_df.index[-1]
        
        # Find the last trading date that we have asset data for
        valid_end_dates = self.assets_df.index[self.assets_df.index > last_weight_date]
        if len(valid_end_dates) == 0:
            self.backtest_end_date = self.assets_df.index[-1]
        else:
            self.backtest_end_date = valid_end_dates[0]
            
        # Trim assets_df to the backtest period
        self.assets_df = self.assets_df[self.backtest_start_date:self.backtest_end_date]
        self.all_dates = self.assets_df.index
        
        # Get rebalance dates within our backtesting period
        mask = (self.weights_df.index >= self.backtest_start_date) & (self.weights_df.index <= self.backtest_end_date)
        self.weights_df = self.weights_df[mask]
        self.rebalance_dates = self.weights_df.index
        
        if len(self.rebalance_dates) == 0:
            raise ValueError("No valid rebalancing dates found within asset data range")
    
    def _calculate_units(self, weights, portfolio_value, prices):
        """
        Calculate the number of units to hold based on target weights and current prices
        
        Parameters:
        weights: pd.Series of target weights
        portfolio_value: float of current portfolio value
        prices: pd.Series of current asset prices
        
        Returns:
        pd.Series of units to hold for each asset
        """
        return (weights * portfolio_value) / prices
    
    def _calculate_current_weights(self, units, prices, portfolio_value):
        """
        Calculate current weights based on units held and current prices
        
        Parameters:
        units: pd.Series of units held
        prices: pd.Series of current asset prices
        portfolio_value: float of current portfolio value
        
        Returns:
        pd.Series of current weights
        """
        asset_values = units * prices
        return asset_values / portfolio_value
    
    def _calculate_turnover_by_asset(self, old_units, new_units, prices):
        """
        Calculate the turnover for each asset when rebalancing
        
        Parameters:
        old_units: pd.Series of current units
        new_units: pd.Series of target units
        prices: pd.Series of current prices
        
        Returns:
        pd.Series of turnover amounts by asset
        """
        unit_changes = np.abs(new_units - old_units)
        return (unit_changes * prices).abs()
    
    def backtest(self):
        """
        Run the backtest
        
        Returns:
        tuple of (net_backtest_df, gross_backtest_df, actual_weights_df) where:
        - net_backtest_df: DataFrame with net-of-cost portfolio values
        - gross_backtest_df: DataFrame with gross-of-cost portfolio values
        - actual_weights_df: DataFrame with actual portfolio weights over time
        """
        # Initialize results
        portfolio_gross = pd.Series(index=self.all_dates, dtype=float)
        portfolio_net = pd.Series(index=self.all_dates, dtype=float)
        units = pd.DataFrame(index=self.all_dates, columns=self.weights_df.columns, dtype=float)
        actual_weights = pd.DataFrame(index=self.all_dates, columns=self.weights_df.columns)
        
        # Initialize portfolio values
        portfolio_gross.iloc[0] = 100
        portfolio_net.iloc[0] = 100
        
        # Initialize first units based on initial weights
        initial_weights = self.weights_df.iloc[0]
        current_units = self._calculate_units(
            initial_weights,
            portfolio_net.iloc[0],
            self.assets_df.iloc[0]
        )
        units.iloc[0] = current_units
        actual_weights.iloc[0] = initial_weights
        
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
            
            # Calculate current portfolio value before rebalancing
            current_portfolio_value = (current_units * current_prices).sum()
            
            # If it's a rebalancing day, update units based on new target weights
            if current_date in rebalance_weights:
                target_weights = pd.Series(rebalance_weights[current_date])
                new_units = self._calculate_units(
                    target_weights,
                    current_portfolio_value,
                    current_prices
                )
                
                # Calculate transaction costs
                turnover_values = self._calculate_turnover_by_asset(
                    current_units,
                    new_units,
                    current_prices
                )
                transaction_costs = sum(
                    turnover_values[asset] * self.costs['transaction_costs'][asset]
                    for asset in turnover_values.index
                ) / current_portfolio_value
                
                current_units = new_units
            else:
                transaction_costs = 0
            
            # Store current units and calculate current weights
            units.loc[current_date] = current_units
            current_weights = self._calculate_current_weights(
                current_units,
                current_prices,
                current_portfolio_value
            )
            actual_weights.loc[current_date] = current_weights
            
            # Calculate portfolio return
            asset_returns = (current_prices - prev_prices) / prev_prices
            portfolio_return = (current_weights * asset_returns).sum()
            
            # Calculate weighted servicing costs
            daily_servicing_cost = sum(
                current_weights[asset] * daily_servicing_costs[asset]
                for asset in current_weights.index
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

    def get_summary_statistics(self):
        """[Previous implementation remains the same]"""
        pass
