import pandas as pd
import numpy as np

class PortfolioBacktester:
    def __init__(self, weights_df, assets_df, costs):
        """
        Initialize the backtester with input data
        
        Parameters:
        weights_df: pd.DataFrame with dates as index and assets as columns, containing desired weights
        assets_df: pd.DataFrame with dates as index and assets as columns, containing asset prices/returns
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
        
        # Convert asset prices to returns if needed
        if not np.allclose(self.assets_df.mean(), 0, atol=0.1):  # Heuristic to detect if data is returns
            self.assets_df = self.assets_df.pct_change()
        
        # Determine the backtest period
        self._determine_backtest_period()
        
        # Sort costs dictionaries to match DataFrame column order
        self.costs = {
            'transaction_costs': {asset: costs['transaction_costs'][asset] 
                                for asset in self.weights_df.columns},
            'servicing_costs': {asset: costs['servicing_costs'][asset] 
                              for asset in self.weights_df.columns}
        }
        
        # Validate costs structure
        required_assets = set(self.weights_df.columns)
        for cost_type in ['transaction_costs', 'servicing_costs']:
            if cost_type not in costs:
                raise ValueError(f"Missing {cost_type} in costs dictionary")
            missing_costs = required_assets - set(costs[cost_type].keys())
            if missing_costs:
                raise ValueError(f"Missing {cost_type} for assets: {missing_costs}")
            
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
            
    def _get_next_trading_date(self, date):
        """Find the next available trading date on or after the given date"""
        future_dates = self.all_dates[self.all_dates >= date]
        return future_dates[0] if len(future_dates) > 0 else None
    
    def _get_drifted_weights(self, start_weights, returns):
        """Calculate how weights drift between rebalancing dates"""
        cumulative_returns = (1 + returns).cumprod()
        drifted_weights = start_weights * cumulative_returns
        return drifted_weights.div(drifted_weights.sum(axis=1), axis=0)
    
    def _calculate_turnover_by_asset(self, old_weights, new_weights):
        """Calculate the turnover for each asset when rebalancing"""
        return pd.Series(np.abs(new_weights - old_weights), index=new_weights.index)
    
    def backtest(self):
        """
        Run the backtest
        
        Returns:
        tuple of (net_backtest_df, gross_backtest_df, actual_weights_df) where:
        - net_backtest_df: DataFrame with net-of-cost portfolio values
        - gross_backtest_df: DataFrame with gross-of-cost portfolio values
        - actual_weights_df: DataFrame with actual portfolio weights over time
        """
        # Initialize results for the backtest period
        portfolio_gross = pd.Series(index=self.all_dates, dtype=float)
        portfolio_net = pd.Series(index=self.all_dates, dtype=float)
        actual_weights = pd.DataFrame(index=self.all_dates, columns=self.weights_df.columns)
        
        # Initialize at the first date with the first weights
        portfolio_gross.iloc[0] = 100
        portfolio_net.iloc[0] = 100
        current_weights = self.weights_df.iloc[0]
        actual_weights.iloc[0] = current_weights
        
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
            
            # Get daily returns
            daily_returns = self.assets_df.loc[current_date]
            
            # If it's a rebalancing day, calculate transaction costs
            if current_date in rebalance_weights:
                new_weights = pd.Series(rebalance_weights[current_date])
                turnover_by_asset = self._calculate_turnover_by_asset(current_weights, new_weights)
                
                # Calculate transaction costs for each asset
                transaction_costs = sum(
                    turnover_by_asset[asset] * self.costs['transaction_costs'][asset]
                    for asset in turnover_by_asset.index
                )
                
                current_weights = new_weights
            else:
                transaction_costs = 0
                # Drift weights according to returns
                current_weights = self._get_drifted_weights(
                    current_weights, 
                    self.assets_df.loc[prev_date:current_date]
                ).iloc[-1]
            
            # Store actual weights
            actual_weights.loc[current_date] = current_weights
            
            # Calculate portfolio returns
            portfolio_return = (current_weights * daily_returns).sum()
            
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
        
        return results_net, results_gross, actual_weights

    def get_summary_statistics(self):
        """[Previous implementation remains the same]"""
        pass
