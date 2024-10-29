import pandas as pd
import numpy as np

class PortfolioBacktester:
    def __init__(self, weights_df, assets_df, costs):
        """[Previous initialization remains the same]"""
        self.initial_portfolio_value = 100
        # [Rest of initialization code remains unchanged]
        pass

    def _calculate_units(self, target_weights, portfolio_value, prices):
        """
        Calculate the number of units to hold based on target weights
        
        Parameters:
        target_weights: pd.Series of target weights
        portfolio_value: float of current portfolio value
        prices: pd.Series of current asset prices
        
        Returns:
        pd.Series of units to hold for each asset
        """
        return (target_weights * portfolio_value) / prices
    
    def _calculate_weights(self, units, prices, portfolio_value):
        """
        Calculate current weights based on units held, current prices, and portfolio value
        
        Parameters:
        units: pd.Series of units held
        prices: pd.Series of current prices
        portfolio_value: float of current portfolio value
        
        Returns:
        pd.Series of current weights
        """
        return (units * prices) / portfolio_value
    
    def backtest(self):
        """Run backtest maintaining constant units between rebalance dates"""
        # Initialize results
        portfolio_value = pd.Series(index=self.all_dates, dtype=float)
        units = pd.DataFrame(index=self.all_dates, columns=self.weights_df.columns, dtype=float)
        weights = pd.DataFrame(index=self.all_dates, columns=self.weights_df.columns)
        
        # Initialize first day
        portfolio_value.iloc[0] = self.initial_portfolio_value
        
        # Calculate initial units from target weights
        initial_weights = self.weights_df.iloc[0]
        current_units = self._calculate_units(
            initial_weights,
            portfolio_value.iloc[0],
            self.assets_df.iloc[0]
        )
        units.iloc[0] = current_units
        
        # Calculate initial weights
        weights.iloc[0] = self._calculate_weights(
            current_units,
            self.assets_df.iloc[0],
            portfolio_value.iloc[0]
        )
        
        # Map rebalance dates to their target weights
        rebalance_weights = self.weights_df.to_dict('index')
        
        # Daily holding costs
        daily_servicing_costs = {
            asset: (1 - cost) ** (1/252) - 1 
            for asset, cost in self.costs['servicing_costs'].items()
        }
        
        for i in range(1, len(self.all_dates)):
            current_date = self.all_dates[i]
            prev_date = self.all_dates[i-1]
            
            # Get prices
            current_prices = self.assets_df.loc[current_date]
            prev_prices = self.assets_df.loc[prev_date]
            
            # Calculate current portfolio value before any rebalancing
            current_portfolio_value = (current_units * current_prices).sum()
            
            # If rebalance day, calculate new units based on target weights
            if current_date in rebalance_weights:
                target_weights = pd.Series(rebalance_weights[current_date])
                new_units = self._calculate_units(
                    target_weights,
                    current_portfolio_value,
                    current_prices
                )
                
                # Calculate transaction costs
                units_change = np.abs(new_units - current_units)
                transaction_value = (units_change * current_prices).sum()
                transaction_costs = sum(
                    units_change[asset] * current_prices[asset] * self.costs['transaction_costs'][asset]
                    for asset in units_change.index
                )
                
                # Update portfolio value for transaction costs
                current_portfolio_value -= transaction_costs
                current_units = new_units
            
            # Store current units
            units.loc[current_date] = current_units
            
            # Calculate and store current weights
            current_weights = self._calculate_weights(
                current_units,
                current_prices,
                current_portfolio_value
            )
            weights.loc[current_date] = current_weights
            
            # Calculate holding costs
            daily_holding_cost = sum(
                current_weights[asset] * daily_servicing_costs[asset]
                for asset in current_weights.index
            ) * current_portfolio_value
            
            # Store portfolio value
            portfolio_value.loc[current_date] = current_portfolio_value - daily_holding_cost
        
        # Create results DataFrame
        results = pd.DataFrame({
            'value': portfolio_value,
            'return': portfolio_value.pct_change()
        })
        
        return results, weights, units

    def get_summary_statistics(self):
        """Calculate summary statistics"""
        results, weights, units = self.backtest()
        
        return {
            'total_return': results['value'].iloc[-1] / results['value'].iloc[0] - 1,
            'annual_return': (1 + results['return']).prod() ** (252/len(results)) - 1,
            'volatility': results['return'].std() * np.sqrt(252),
            'max_drawdown': (results['value'] / results['value'].cummax() - 1).min(),
            'final_weights_sum': weights.iloc[-1].sum(),
            'average_weights_sum': weights.sum(axis=1).mean()
        }
