import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def forecast_demand(sales_data, days_ahead=14):
    """
    Forecast future demand based on historical sales data.
    
    Args:
        sales_data: DataFrame containing historical sales data
        days_ahead: Number of days to forecast ahead
        
    Returns:
        DataFrame with forecasted demand for the next X days
    """
    # In a real implementation, we would use time series models like ARIMA, Prophet, or LSTM
    # For this demo, we'll generate simulated forecast data
    
    # Generate dates for the forecast period using March 30, 2025 as reference
    today = datetime(2025, 3, 30)  # Fixed date (March 30, 2025)
    forecast_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_ahead)]
    
    # Create basic patterns with weekly seasonality
    base_demand = 100
    day_of_week_factors = {
        0: 0.8,  # Monday
        1: 0.9,  # Tuesday
        2: 1.0,  # Wednesday
        3: 1.1,  # Thursday
        4: 1.3,  # Friday
        5: 1.4,  # Saturday
        6: 1.2   # Sunday
    }
    
    # Generate forecasted demand with seasonal patterns and random noise
    forecasted_values = []
    for i, date_str in enumerate(forecast_dates):
        date = datetime.strptime(date_str, '%Y-%m-%d')
        day_factor = day_of_week_factors[date.weekday()]
        
        # Add some trend and random noise
        trend_factor = 1 + (i * 0.01)  # Small upward trend
        noise = np.random.normal(0, 0.05)  # Small random variations
        
        # Calculate demand for this day
        demand = base_demand * day_factor * trend_factor * (1 + noise)
        forecasted_values.append(round(demand, 2))
    
    # Create a DataFrame with the forecast
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecasted_demand': forecasted_values
    })
    
    # Set date as index for better chart display
    forecast_df.set_index('date', inplace=True)
    
    return forecast_df

def predict_waste(inventory_data, sales_data, threshold_days=5):
    """
    Predict potential waste based on inventory expiry dates and forecasted demand.
    
    Args:
        inventory_data: List of inventory items with expiry dates
        sales_data: DataFrame containing historical sales data
        threshold_days: Number of days to consider for waste prediction
        
    Returns:
        DataFrame with waste predictions by category
    """
    # In a real implementation, we would:
    # 1. Use the forecasted demand to estimate usage rates
    # 2. Compare with inventory expiry dates
    # 3. Calculate potential waste
    
    # For this demo, we'll generate simulated waste prediction data
    
    # Define food categories
    categories = ['Vegetables', 'Fruits', 'Protein', 'Dairy', 'Grains', 'Prepared Foods']
    
    # Generate waste predictions with realistic patterns
    waste_values = [
        round(random.uniform(4, 8), 1),   # Vegetables - higher waste
        round(random.uniform(3, 7), 1),   # Fruits - higher waste
        round(random.uniform(1, 3), 1),   # Protein - lower waste
        round(random.uniform(2, 4), 1),   # Dairy - medium waste
        round(random.uniform(0.5, 2), 1), # Grains - low waste
        round(random.uniform(2, 5), 1)    # Prepared Foods - medium waste
    ]
    
    # Create a DataFrame with the waste predictions
    waste_df = pd.DataFrame({
        'category': categories,
        'predicted_waste_kg': waste_values
    })
    
    # Set category as index for better chart display
    waste_df.set_index('category', inplace=True)
    
    return waste_df

def analyze_stock_levels(inventory_data, forecasted_demand):
    """
    Analyze current stock levels against forecasted demand to identify over/under stocking.
    
    Args:
        inventory_data: List of inventory items with quantities
        forecasted_demand: DataFrame with forecasted demand
        
    Returns:
        Dictionary with stock status analyses
    """
    # In a production environment, this would analyze each item category
    # For this demo, we'll return simulated results
    
    stock_analysis = {
        'overstocked_items': [
            {'category': 'Vegetables', 'item': 'Potatoes', 'excess_qty': 8, 'excess_value': 12.00},
            {'category': 'Dairy', 'item': 'Butter', 'excess_qty': 5, 'excess_value': 22.50}
        ],
        'understocked_items': [
            {'category': 'Protein', 'item': 'Chicken Breast', 'shortage_qty': 3, 'potential_loss': 45.00},
            {'category': 'Prepared Foods', 'item': 'Pasta Sauce', 'shortage_qty': 2, 'potential_loss': 18.00}
        ],
        'optimal_items': 12,  # Number of items with appropriate stock levels
        'reorder_recommendations': [
            {'item': 'Chicken Breast', 'recommended_qty': 5},
            {'item': 'Pasta Sauce', 'recommended_qty': 3},
            {'item': 'Lettuce', 'recommended_qty': 4}
        ]
    }
    
    return stock_analysis
