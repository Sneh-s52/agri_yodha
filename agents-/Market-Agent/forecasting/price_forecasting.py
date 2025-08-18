# market_agent/forecasting/price_forecasting.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PriceForecastingTool:
    """
    A comprehensive price forecasting tool that uses multiple statistical models
    and can work with limited data for agricultural commodity prices.
    """

    def __init__(self):
        """Initialize the forecasting tool with default parameters."""
        self.models = {}
        self.scaler = StandardScaler()
        self.forecast_history = {}
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'modal_price') -> pd.DataFrame:
        """
        Prepare and clean data for forecasting.
        
        Args:
            df: Raw DataFrame with price data
            target_column: Name of the price column
            
        Returns:
            Cleaned and prepared DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for forecasting")
            return df
            
        try:
            # Ensure we have the required columns
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found in DataFrame")
                return df
                
            # Remove any rows with missing prices
            df_clean = df.dropna(subset=[target_column])
            
            if df_clean.empty:
                logger.warning("No valid price data after cleaning")
                return df_clean
                
            # Handle duplicate index labels
            if df_clean.index.duplicated().any():
                logger.warning("Duplicate index labels found, keeping last occurrence")
                df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
                
            # Convert price to numeric with better error handling
            try:
                # First, try to convert directly
                df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors='coerce')
            except Exception as e:
                logger.warning(f"Direct numeric conversion failed: {e}")
                # Try to clean string data
                df_clean[target_column] = df_clean[target_column].astype(str).str.replace(',', '').str.replace('₹', '').str.replace('Rs', '').str.strip()
                df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors='coerce')
                
            # Remove rows with invalid prices
            df_clean = df_clean.dropna(subset=[target_column])
            
            # Remove outliers (prices that are too high or too low)
            if len(df_clean) > 10:
                Q1 = df_clean[target_column].quantile(0.25)
                Q3 = df_clean[target_column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Filter out outliers
                df_clean = df_clean[(df_clean[target_column] >= lower_bound) & 
                                   (df_clean[target_column] <= upper_bound)]
                
            if df_clean.empty:
                logger.warning("No valid price data after outlier removal")
                return df_clean
                
            # Sort by date
            df_clean = df_clean.sort_index()
            
            # Resample to daily frequency and forward fill missing values
            try:
                df_resampled = df_clean.resample('D').ffill()
            except Exception as e:
                logger.warning(f"Resampling failed, using original data: {e}")
                df_resampled = df_clean
                
            # If we have very few data points, interpolate to create more
            if len(df_resampled) < 30:
                df_resampled = df_resampled.interpolate(method='linear')
                
            # Ensure we have enough data for forecasting
            if len(df_resampled) < 7:
                logger.warning(f"Insufficient data points: {len(df_resampled)}")
                return df_resampled
                
            logger.info(f"Prepared data: {len(df_resampled)} data points for forecasting")
            return df_resampled
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return df

    def simple_moving_average_forecast(self, df: pd.DataFrame, target_column: str = 'modal_price', 
                                     window: int = 7, forecast_days: int = 7) -> Dict[str, Any]:
        """
        Simple moving average forecasting for short-term predictions.
        
        Args:
            df: Prepared DataFrame
            target_column: Price column name
            window: Moving average window size
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            if len(df) < window:
                return {"error": f"Insufficient data. Need at least {window} data points."}
                
            # Calculate moving average
            ma = df[target_column].rolling(window=window).mean()
            
            # Use the last moving average value for forecasting
            last_ma = ma.iloc[-1]
            
            # Generate forecast
            forecast_values = [last_ma] * forecast_days
            forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), 
                                         periods=forecast_days, freq='D')
            
            # Calculate confidence interval (simple approach)
            std_dev = df[target_column].rolling(window=window).std().iloc[-1]
            confidence_interval = 1.96 * std_dev  # 95% confidence
            
            return {
                "method": "Simple Moving Average",
                "forecast_dates": [d.isoformat() for d in forecast_dates],
                "forecast_values": forecast_values,
                "confidence_interval": confidence_interval,
                "last_actual_price": df[target_column].iloc[-1],
                "last_ma_value": last_ma,
                "window_size": window,
                "data_points_used": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in simple moving average forecast: {e}")
            return {"error": f"Forecasting error: {str(e)}"}

    def exponential_smoothing_forecast(self, df: pd.DataFrame, target_column: str = 'modal_price',
                                    forecast_days: int = 7) -> Dict[str, Any]:
        """
        Exponential smoothing forecast using Holt-Winters method.
        
        Args:
            df: Prepared DataFrame
            target_column: Price column name
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            if len(df) < 10:
                return {"error": "Insufficient data for exponential smoothing. Need at least 10 data points."}
                
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                df[target_column],
                trend='add',
                seasonal='add',
                seasonal_periods=min(7, len(df) // 2)  # Weekly seasonality if possible
            )
            
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(forecast_days)
            
            # Calculate confidence intervals (simplified approach)
            # Note: statsmodels ExponentialSmoothing doesn't provide built-in confidence intervals
            # So we'll calculate a simple confidence interval based on historical volatility
            historical_std = df[target_column].std()
            confidence_interval = 1.96 * historical_std  # 95% confidence
            
            return {
                "method": "Exponential Smoothing (Holt-Winters)",
                "forecast_dates": [d.isoformat() for d in forecast.index],
                "forecast_values": forecast.values.tolist(),
                "confidence_intervals": {
                    "lower": [f - confidence_interval for f in forecast.values],
                    "upper": [f + confidence_interval for f in forecast.values]
                },
                "last_actual_price": df[target_column].iloc[-1],
                "model_aic": fitted_model.aic,
                "data_points_used": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing forecast: {e}")
            return {"error": f"Forecasting error: {str(e)}"}

    def arima_forecast(self, df: pd.DataFrame, target_column: str = 'modal_price',
                      forecast_days: int = 7) -> Dict[str, Any]:
        """
        ARIMA forecasting for time series data.
        
        Args:
            df: Prepared DataFrame
            target_column: Price column name
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            if len(df) < 20:
                return {"error": "Insufficient data for ARIMA. Need at least 20 data points."}
                
            # Try different ARIMA parameters for best fit
            best_aic = float('inf')
            best_model = None
            best_params = None
            
            # Grid search for best ARIMA parameters
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(df[target_column], order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                                best_params = (p, d, q)
                        except:
                            continue
            
            if best_model is None:
                return {"error": "Could not fit ARIMA model with any parameters."}
                
            # Generate forecast
            forecast = best_model.forecast(steps=forecast_days)
            
            # Calculate confidence intervals
            forecast_conf = best_model.get_forecast(steps=forecast_days)
            conf_int = forecast_conf.conf_int()
            
            return {
                "method": f"ARIMA{best_params}",
                "forecast_dates": [d.isoformat() for d in forecast.index],
                "forecast_values": forecast.values.tolist(),
                "confidence_intervals": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                },
                "last_actual_price": df[target_column].iloc[-1],
                "model_aic": best_aic,
                "arima_params": best_params,
                "data_points_used": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecast: {e}")
            return {"error": f"Forecasting error: {str(e)}"}

    def ensemble_forecast(self, df: pd.DataFrame, target_column: str = 'modal_price',
                         forecast_days: int = 7) -> Dict[str, Any]:
        """
        Ensemble forecasting combining multiple methods for robust predictions.
        
        Args:
            df: Prepared DataFrame
            target_column: Price column name
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with ensemble forecast results
        """
        try:
            forecasts = []
            weights = []
            
            # Get forecasts from different methods
            ma_forecast = self.simple_moving_average_forecast(df, target_column, forecast_days=forecast_days)
            if "error" not in ma_forecast:
                forecasts.append(ma_forecast)
                weights.append(0.3)  # Lower weight for simple methods
                
            es_forecast = self.exponential_smoothing_forecast(df, target_column, forecast_days=forecast_days)
            if "error" not in es_forecast:
                forecasts.append(es_forecast)
                weights.append(0.4)  # Higher weight for statistical methods
                
            arima_forecast = self.arima_forecast(df, target_column, forecast_days=forecast_days)
            if "error" not in arima_forecast:
                forecasts.append(arima_forecast)
                weights.append(0.3)  # Equal weight for ARIMA
                
            if not forecasts:
                return {"error": "No forecasting methods could be applied successfully."}
                
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Combine forecasts using weighted average
            ensemble_forecast = []
            for day in range(forecast_days):
                day_forecast = 0
                for i, forecast in enumerate(forecasts):
                    if day < len(forecast["forecast_values"]):
                        day_forecast += forecast["forecast_values"][day] * weights[i]
                ensemble_forecast.append(day_forecast)
                
            # Calculate ensemble confidence interval
            all_forecasts = []
            for forecast in forecasts:
                if "forecast_values" in forecast:
                    all_forecasts.extend(forecast["forecast_values"])
                    
            if all_forecasts:
                ensemble_std = np.std(all_forecasts)
                confidence_interval = 1.96 * ensemble_std
            else:
                confidence_interval = 0
                
            return {
                "method": "Ensemble (Weighted Average)",
                "forecast_dates": forecasts[0]["forecast_dates"] if forecasts else [],
                "forecast_values": ensemble_forecast,
                "confidence_interval": confidence_interval,
                "last_actual_price": df[target_column].iloc[-1] if not df.empty else None,
                "methods_used": [f["method"] for f in forecasts],
                "weights": weights,
                "data_points_used": len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble forecast: {e}")
            return {"error": f"Ensemble forecasting error: {str(e)}"}

    def analyze_trends(self, df: pd.DataFrame, target_column: str = 'modal_price') -> Dict[str, Any]:
        """
        Analyze price trends and patterns in the data.
        
        Args:
            df: Prepared DataFrame
            target_column: Price column name
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            if df.empty or len(df) < 2:
                return {"error": "Insufficient data for trend analysis."}
                
            prices = df[target_column]
            
            # Basic statistics
            stats = {
                "mean_price": float(prices.mean()),
                "median_price": float(prices.median()),
                "std_price": float(prices.std()),
                "min_price": float(prices.min()),
                "max_price": float(prices.max()),
                "price_range": float(prices.max() - prices.min()),
                "data_points": len(prices)
            }
            
            # Trend analysis
            if len(prices) >= 2:
                # Linear trend
                x = np.arange(len(prices))
                slope, intercept = np.polyfit(x, prices, 1)
                trend_strength = np.corrcoef(x, prices)[0, 1]
                
                # Price change
                first_price = prices.iloc[0]
                last_price = prices.iloc[-1]
                total_change = last_price - first_price
                percent_change = (total_change / first_price) * 100 if first_price != 0 else 0
                
                # Recent trend (last 30% of data)
                recent_start = max(0, int(len(prices) * 0.7))
                recent_prices = prices.iloc[recent_start:]
                if len(recent_prices) >= 2:
                    recent_x = np.arange(len(recent_prices))
                    recent_slope, _ = np.polyfit(recent_x, recent_prices, 1)
                    recent_trend = "increasing" if recent_slope > 0 else "decreasing"
                else:
                    recent_trend = "insufficient data"
                    
                trends = {
                    "overall_trend": "increasing" if slope > 0 else "decreasing",
                    "trend_strength": float(abs(trend_strength)),
                    "trend_strength_label": "strong" if abs(trend_strength) > 0.7 else "moderate" if abs(trend_strength) > 0.3 else "weak",
                    "total_price_change": float(total_change),
                    "percent_change": float(percent_change),
                    "recent_trend": recent_trend,
                    "trend_slope": float(slope)
                }
            else:
                trends = {"error": "Insufficient data for trend calculation"}
                
            # Volatility analysis
            if len(prices) >= 2:
                price_changes = prices.diff().dropna()
                volatility = float(price_changes.std())
                volatility_label = "high" if volatility > stats["mean_price"] * 0.1 else "moderate" if volatility > stats["mean_price"] * 0.05 else "low"
                
                volatility_analysis = {
                    "volatility": volatility,
                    "volatility_label": volatility_label,
                    "max_daily_change": float(price_changes.max()),
                    "min_daily_change": float(price_changes.min())
                }
            else:
                volatility_analysis = {"error": "Insufficient data for volatility analysis"}
                
            return {
                "statistics": stats,
                "trends": trends,
                "volatility": volatility_analysis,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {"error": f"Trend analysis error: {str(e)}"}

    def generate_forecast_report(self, commodity: str, apmc: str, df: pd.DataFrame,
                               forecast_days: int = 7) -> Dict[str, Any]:
        """
        Generate a comprehensive forecast report for a commodity.
        
        Args:
            commodity: Name of the commodity
            apmc: Name of the APMC
            df: Prepared price data
            forecast_days: Number of days to forecast
            
        Returns:
            Comprehensive forecast report
        """
        try:
            if df.empty:
                return {"error": "No data available for forecasting"}
                
            # Prepare data
            df_prepared = self.prepare_data(df)
            
            if df_prepared.empty:
                return {"error": "Data preparation failed"}
                
            # Generate forecasts using different methods
            forecasts = {}
            
            # Simple moving average
            ma_result = self.simple_moving_average_forecast(df_prepared, forecast_days=forecast_days)
            if "error" not in ma_result:
                forecasts["moving_average"] = ma_result
                
            # Exponential smoothing
            es_result = self.exponential_smoothing_forecast(df_prepared, forecast_days=forecast_days)
            if "error" not in es_result:
                forecasts["exponential_smoothing"] = es_result
                
            # ARIMA
            arima_result = self.arima_forecast(df_prepared, forecast_days=forecast_days)
            if "error" not in arima_result:
                forecasts["arima"] = arima_result
                
            # Ensemble forecast
            ensemble_result = self.ensemble_forecast(df_prepared, forecast_days=forecast_days)
            if "error" not in ensemble_result:
                forecasts["ensemble"] = ensemble_result
                
            # Trend analysis
            trends = self.analyze_trends(df_prepared)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(forecasts, trends, commodity, apmc)
            
            report = {
                "commodity": commodity,
                "apmc": apmc,
                "report_date": datetime.now().isoformat(),
                "forecast_period": f"{forecast_days} days",
                "data_summary": {
                    "total_data_points": len(df_prepared),
                    "date_range": {
                        "start": df_prepared.index[0].isoformat() if not df_prepared.empty else None,
                        "end": df_prepared.index[-1].isoformat() if not df_prepared.empty else None
                    },
                    "last_price": float(df_prepared.iloc[-1]['modal_price']) if not df_prepared.empty else None
                },
                "forecasts": forecasts,
                "trend_analysis": trends,
                "recommendations": recommendations,
                "methodology": {
                    "description": "Multiple forecasting methods combined for robust predictions",
                    "methods_used": list(forecasts.keys()),
                    "data_quality": "good" if len(df_prepared) >= 30 else "limited" if len(df_prepared) >= 10 else "poor"
                }
            }
            
            # Store forecast history
            self.forecast_history[f"{commodity}_{apmc}"] = {
                "timestamp": datetime.now().isoformat(),
                "forecast": report
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating forecast report: {e}")
            return {"error": f"Report generation error: {str(e)}"}

    def _generate_recommendations(self, forecasts: Dict, trends: Dict, 
                                commodity: str, apmc: str) -> List[str]:
        """Generate actionable recommendations based on forecasts and trends."""
        recommendations = []
        
        try:
            # Get ensemble forecast if available
            ensemble = forecasts.get("ensemble", {})
            if "forecast_values" in ensemble and len(ensemble["forecast_values"]) > 0:
                current_price = ensemble.get("last_actual_price", 0)
                forecast_price = ensemble["forecast_values"][0]  # Next day forecast
                
                if current_price and forecast_price:
                    price_change = forecast_price - current_price
                    percent_change = (price_change / current_price) * 100
                    
                    if percent_change > 5:
                        recommendations.append(f"Strong upward price movement expected for {commodity} at {apmc}. Consider strategic procurement.")
                    elif percent_change > 2:
                        recommendations.append(f"Moderate price increase expected. Monitor market conditions closely.")
                    elif percent_change < -5:
                        recommendations.append(f"Significant price decline expected. Evaluate selling strategies.")
                    elif percent_change < -2:
                        recommendations.append(f"Moderate price decrease expected. Consider holding inventory.")
                    else:
                        recommendations.append(f"Stable price movement expected. Maintain current market position.")
                        
            # Add trend-based recommendations
            if "trends" in trends and "error" not in trends["trends"]:
                trend_info = trends["trends"]
                if trend_info.get("trend_strength_label") == "strong":
                    recommendations.append(f"Strong {trend_info.get('overall_trend')} trend detected. Align strategies accordingly.")
                    
            # Add volatility-based recommendations
            if "volatility" in trends and "error" not in trends["volatility"]:
                vol_info = trends["volatility"]
                if vol_info.get("volatility_label") == "high":
                    recommendations.append("High price volatility detected. Implement risk management strategies.")
                    
            # Add data quality recommendations
            if len(forecasts) < 2:
                recommendations.append("Limited forecasting methods available due to data constraints. Consider collecting more historical data.")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to data limitations.")
            
        return recommendations

    def get_forecast_history(self, commodity: str = None, apmc: str = None) -> Dict[str, Any]:
        """Get forecast history for analysis."""
        if commodity and apmc:
            key = f"{commodity}_{apmc}"
            return self.forecast_history.get(key, {})
        return self.forecast_history


# Factory function for easy instantiation
def create_forecasting_tool() -> PriceForecastingTool:
    """Create and return a PriceForecastingTool instance."""
    return PriceForecastingTool()


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    prices = [100 + np.random.normal(0, 5) + i*0.1 for i in range(len(dates))]
    
    sample_df = pd.DataFrame({
        'modal_price': prices
    }, index=dates)
    
    # Test the forecasting tool
    forecaster = PriceForecastingTool()
    
    # Generate forecast report
    report = forecaster.generate_forecast_report("Wheat", "Delhi APMC", sample_df, forecast_days=7)
    
    print("Forecast Report:")
    print(f"Commodity: {report.get('commodity')}")
    print(f"APMC: {report.get('apmc')}")
    print(f"Data Points: {report.get('data_summary', {}).get('total_data_points')}")
    
    if "forecasts" in report:
        for method, forecast in report["forecasts"].items():
            print(f"\n{method.upper()} Forecast:")
            if "forecast_values" in forecast:
                print(f"Next 7 days: {forecast['forecast_values']}")
                
    if "recommendations" in report:
        print(f"\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"- {rec}")
