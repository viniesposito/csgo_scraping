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
        """
        return (target_weights * portfolio_value) / prices
    
    def _calculate_weights(self, units, prices, portfolio_value):
        """
        Calculate current weights based on units held and current prices
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
        weights.iloc[0] = initial_weights  # Initial weights are the target weights
        
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
            
            # Get prices and calculate returns
            current_prices = self.assets_df.loc[current_date]
            prev_prices = self.assets_df.loc[prev_date]
            asset_returns = (current_prices - prev_prices) / prev_prices
            
            # Calculate current weights before any rebalancing
            prev_weights = weights.loc[prev_date]
            
            # Calculate portfolio return
            portfolio_return = (prev_weights * asset_returns).sum()
            
            # Calculate servicing costs based on previous weights
            daily_servicing_cost = sum(
                prev_weights[asset] * daily_servicing_costs[asset]
                for asset in prev_weights.index
            )
            
            # Update portfolio value based on return and servicing costs
            portfolio_value.loc[current_date] = portfolio_value.loc[prev_date] * (1 + portfolio_return - daily_servicing_cost)
            
            # If rebalance day, calculate new units based on target weights
            if current_date in rebalance_weights:
                target_weights = pd.Series(rebalance_weights[current_date])
                new_units = self._calculate_units(
                    target_weights,
                    portfolio_value.loc[current_date],
                    current_prices
                )
                
                # Calculate transaction costs
                old_position_values = current_units * current_prices
                new_position_values = new_units * current_prices
                traded_values = np.abs(new_position_values - old_position_values)
                
                transaction_costs = sum(
                    traded_values[asset] * self.costs['transaction_costs'][asset]
                    for asset in traded_values.index
                ) / portfolio_value.loc[current_date]
                
                # Apply transaction costs to portfolio value
                portfolio_value.loc[current_date] *= (1 - transaction_costs)
                
                # Update units and calculate new weights
                current_units = new_units
                current_weights = target_weights
                
            else:
                # On non-rebalancing days, calculate weights based on returns
                current_weights = prev_weights * (1 + asset_returns) / (1 + portfolio_return)
            
            # Store current units and weights
            units.loc[current_date] = current_units
            weights.loc[current_date] = current_weights
        
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
