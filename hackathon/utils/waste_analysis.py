import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def analyze_waste(waste_data):
    """
    Analyze waste data to identify trends and patterns.
    
    Args:
        waste_data: Historical waste data records
        
    Returns:
        Dictionary with waste analysis metrics and insights
    """
    # In a real implementation, we would:
    # 1. Process historical waste data
    # 2. Identify patterns, trends, and seasonality
    # 3. Generate actionable insights
    
    # For this demo, we'll return simulated waste analysis
    
    # Calculate overall waste metrics
    total_waste_kg = sum(item['quantity_kg'] for item in waste_data) if waste_data else 25.5
    # Convert USD to INR (multiplier of 83)
    total_waste_value = sum(item['value'] for item in waste_data) if waste_data else 342.75 * 83
    
    # Calculate waste by category
    categories = ["Vegetables", "Fruits", "Protein", "Dairy", "Grains", "Prepared Foods"]
    waste_by_category = {}
    
    for category in categories:
        category_items = [item for item in waste_data if item['category'] == category] if waste_data else []
        category_waste_kg = sum(item['quantity_kg'] for item in category_items) if category_items else random.uniform(2, 8)
        category_waste_value = sum(item['value'] for item in category_items) if category_items else category_waste_kg * random.uniform(8, 15) * 83  # Convert USD to INR
        
        waste_by_category[category] = {
            'quantity_kg': round(category_waste_kg, 2),
            'value': round(category_waste_value, 2),
            'percentage': round(category_waste_kg / total_waste_kg * 100 if total_waste_kg > 0 else 0, 1)
        }
    
    # Calculate waste by reason
    waste_reasons = ["Expired", "Over-production", "Damaged", "Trim waste", "Quality issues"]
    waste_by_reason = {}
    
    for reason in waste_reasons:
        reason_items = [item for item in waste_data if item['reason'] == reason] if waste_data else []
        reason_waste_kg = sum(item['quantity_kg'] for item in reason_items) if reason_items else random.uniform(1, 6)
        reason_waste_value = sum(item['value'] for item in reason_items) if reason_items else reason_waste_kg * random.uniform(10, 18) * 83  # Convert USD to INR
        
        waste_by_reason[reason] = {
            'quantity_kg': round(reason_waste_kg, 2),
            'value': round(reason_waste_value, 2),
            'percentage': round(reason_waste_kg / total_waste_kg * 100 if total_waste_kg > 0 else 0, 1)
        }
    
    # Generate waste trends (simulated time series)
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    
    # Create base trend with weekly pattern and slight downward trend
    base_trend = [4 + (0.2 * (i % 7)) - (i * 0.03) for i in range(30)]
    
    # Add random noise
    waste_trend = [max(0, value + random.uniform(-0.5, 0.5)) for value in base_trend]
    
    # Create trend data structure
    trend_data = {dates[i]: round(waste_trend[i], 2) for i in range(len(dates))}
    
    # Create insights based on the analysis
    insights = [
        "Vegetable waste is the largest category by weight, primarily due to spoilage and over-ordering.",
        "Thursday and Friday show higher waste levels, suggesting weekend prep may be contributing to excess.",
        "Prepared foods waste has decreased by 18% since implementing portioning guidelines.",
        "Trimming waste accounts for 22% of total waste, indicating potential for improved cutting techniques or repurposing.",
        "The highest value waste comes from protein items, particularly seafood and specialty meats."
    ]
    
    # Create recommendations
    recommendations = [
        "Reduce standing vegetable orders by 15% and increase order frequency.",
        "Implement a 'Thursday special' using ingredients approaching expiry date.",
        "Conduct staff training on efficient prep techniques to reduce trim waste.",
        "Create a standardized process for repurposing trim and off-cuts.",
        "Adjust par levels for items with highest waste percentages."
    ]
    
    # Compile the complete analysis
    waste_analysis = {
        'total_waste_kg': round(total_waste_kg, 2),
        'total_waste_value': round(total_waste_value, 2),
        'waste_by_category': waste_by_category,
        'waste_by_reason': waste_by_reason,
        'trend_data': trend_data,
        'insights': insights,
        'recommendations': recommendations
    }
    
    return waste_analysis

def calculate_waste_metrics(inventory_data):
    """
    Calculate the financial impact of potential waste based on inventory.
    
    Args:
        inventory_data: Current inventory with expiry dates
        
    Returns:
        Float representing the total potential waste value
    """
    # In a real implementation, we would:
    # 1. Identify items likely to expire before use
    # 2. Calculate their value
    
    # For this demo, we'll return a simulated value
    total_value = 0
    
    # Calculate potential waste value using inventory data
    if inventory_data:
        for item in inventory_data:
            # Check if item has expiry_date and is soon expiring
            if 'expiry_date' in item:
                try:
                    expiry_date = pd.to_datetime(item['expiry_date'])
                    days_until_expiry = (expiry_date - pd.Timestamp.now()).days
                    
                    # If expiring within 3 days, add to potential waste
                    if days_until_expiry <= 3:
                        item_value = item.get('value', 0) or item.get('price', 0) * item.get('quantity', 0)
                        # The closer to expiry, the more likely to become waste
                        waste_probability = max(0, 1 - (days_until_expiry / 3))
                        total_value += item_value * waste_probability
                except:
                    # If date parsing fails, skip this item
                    pass
    
    # If no waste was calculated or no inventory data, return a reasonable default
    if total_value == 0:
        total_value = random.uniform(180, 350) * 83  # Convert USD to INR
    
    return round(total_value, 2)

def generate_waste_heatmap_data():
    """
    Generate data for a waste heatmap visualization.
    
    Returns:
        Dictionary mapping kitchen areas to waste levels
    """
    # In a real implementation, we would analyze actual waste tracking data
    # For this demo, we'll return simulated heatmap data
    
    kitchen_areas = [
        "Prep Station", "Grill Station", "Fry Station", "Salad Station",
        "Dessert Station", "Main Cooler", "Vegetable Cooler", "Meat Cooler",
        "Dish Station", "Plating Area", "Bar", "Receiving Area"
    ]
    
    # Create waste levels with some areas having consistently higher waste
    high_waste_areas = ["Prep Station", "Salad Station", "Vegetable Cooler"]
    medium_waste_areas = ["Grill Station", "Dish Station", "Plating Area"]
    
    heatmap_data = {}
    for area in kitchen_areas:
        if area in high_waste_areas:
            waste_level = random.uniform(0.6, 1.0)
        elif area in medium_waste_areas:
            waste_level = random.uniform(0.3, 0.6)
        else:
            waste_level = random.uniform(0.1, 0.3)
        
        heatmap_data[area] = round(waste_level, 2)
    
    return heatmap_data
