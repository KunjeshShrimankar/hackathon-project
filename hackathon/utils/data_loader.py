import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

def load_inventory_data(force_regenerate=True):  # Default changed to True to always regenerate with 2025 dates
    """
    Load inventory data from a CSV file or create sample data if file doesn't exist.
    
    Args:
        force_regenerate: If True, regenerate sample data even if file exists
    
    Returns:
        List of dictionaries containing inventory data
    """
    print(f"Loading inventory data with force_regenerate={force_regenerate}")
    # Check if the file exists and not forcing regeneration
    if os.path.exists('data/sample_inventory.csv') and not force_regenerate:
        print("Loading existing inventory data from CSV")
        # Load the data from the CSV file
        try:
            df = pd.read_csv('data/sample_inventory.csv')
            inventory_data = df.to_dict('records')
            return inventory_data
        except Exception as e:
            print(f"Error loading inventory data: {e}")
            # Fall back to generating sample data
            print("Generating sample inventory data due to error")
            return generate_sample_inventory()
    else:
        # Generate sample data
        print("Regenerating sample inventory data with 2025 dates")
        inventory_data = generate_sample_inventory()
        # Save to CSV for future use
        save_inventory_data(inventory_data)
        return inventory_data

def load_sales_data(force_regenerate=True):  # Default changed to True to always regenerate with 2025 dates
    """
    Load sales data from a CSV file or create sample data if file doesn't exist.
    
    Args:
        force_regenerate: If True, regenerate sample data even if file exists
    
    Returns:
        DataFrame containing sales data
    """
    print(f"Loading sales data with force_regenerate={force_regenerate}")
    # Check if the file exists and not forcing regeneration
    if os.path.exists('data/sample_sales.csv') and not force_regenerate:
        print("Loading existing sales data from CSV")
        # Load the data from the CSV file
        try:
            df = pd.read_csv('data/sample_sales.csv')
            return df
        except Exception as e:
            print(f"Error loading sales data: {e}")
            # Fall back to generating sample data
            print("Generating sample sales data due to error")
            return generate_sample_sales()
    else:
        # Generate sample data
        print("Regenerating sample sales data with 2025 dates")
        sales_data = generate_sample_sales()
        # Save to CSV for future use
        save_sales_data(sales_data)
        return sales_data

def generate_sample_inventory():
    """
    Generate sample inventory data for demonstration purposes.
    
    Returns:
        List of dictionaries containing inventory data
    """
    # Define categories and sample items in each category with prices in Indian Rupees (₹)
    categories = {
        'Vegetables': [
            {'name': 'Tomatoes', 'unit': 'kg', 'price': 80, 'shelf_life': 7},
            {'name': 'Lettuce', 'unit': 'head', 'price': 50, 'shelf_life': 5},
            {'name': 'Bell Peppers', 'unit': 'kg', 'price': 100, 'shelf_life': 10},
            {'name': 'Onions', 'unit': 'kg', 'price': 40, 'shelf_life': 14},
            {'name': 'Carrots', 'unit': 'kg', 'price': 45, 'shelf_life': 21},
            {'name': 'Potatoes', 'unit': 'kg', 'price': 30, 'shelf_life': 30},
            {'name': 'Zucchini', 'unit': 'kg', 'price': 70, 'shelf_life': 7},
            {'name': 'Eggplant', 'unit': 'kg', 'price': 80, 'shelf_life': 7},
        ],
        'Fruits': [
            {'name': 'Apples', 'unit': 'kg', 'price': 160, 'shelf_life': 14},
            {'name': 'Bananas', 'unit': 'kg', 'price': 60, 'shelf_life': 7},
            {'name': 'Lemons', 'unit': 'kg', 'price': 120, 'shelf_life': 14},
            {'name': 'Oranges', 'unit': 'kg', 'price': 160, 'shelf_life': 14},
            {'name': 'Strawberries', 'unit': 'punnet', 'price': 250, 'shelf_life': 5},
            {'name': 'Blueberries', 'unit': 'punnet', 'price': 300, 'shelf_life': 7},
        ],
        'Protein': [
            {'name': 'Chicken Breast', 'unit': 'kg', 'price': 350, 'shelf_life': 4},
            {'name': 'Ground Beef', 'unit': 'kg', 'price': 400, 'shelf_life': 3},
            {'name': 'Salmon Fillets', 'unit': 'kg', 'price': 800, 'shelf_life': 3},
            {'name': 'Tofu', 'unit': 'block', 'price': 150, 'shelf_life': 10},
            {'name': 'Eggs', 'unit': 'dozen', 'price': 120, 'shelf_life': 21},
            {'name': 'Shrimp', 'unit': 'kg', 'price': 600, 'shelf_life': 4},
            {'name': 'Beef Steak', 'unit': 'kg', 'price': 650, 'shelf_life': 5},
        ],
        'Dairy': [
            {'name': 'Milk', 'unit': 'liter', 'price': 75, 'shelf_life': 10},
            {'name': 'Cheddar Cheese', 'unit': 'kg', 'price': 550, 'shelf_life': 30},
            {'name': 'Butter', 'unit': '250g', 'price': 130, 'shelf_life': 30},
            {'name': 'Greek Yogurt', 'unit': 'kg', 'price': 220, 'shelf_life': 14},
            {'name': 'Cream Cheese', 'unit': '250g', 'price': 180, 'shelf_life': 21},
            {'name': 'Heavy Cream', 'unit': 'liter', 'price': 250, 'shelf_life': 14},
        ],
        'Grains': [
            {'name': 'Rice', 'unit': 'kg', 'price': 85, 'shelf_life': 180},
            {'name': 'Pasta', 'unit': 'kg', 'price': 120, 'shelf_life': 180},
            {'name': 'Bread', 'unit': 'loaf', 'price': 70, 'shelf_life': 7},
            {'name': 'Flour', 'unit': 'kg', 'price': 50, 'shelf_life': 180},
            {'name': 'Quinoa', 'unit': 'kg', 'price': 350, 'shelf_life': 180},
        ],
        'Herbs & Spices': [
            {'name': 'Basil', 'unit': 'bunch', 'price': 60, 'shelf_life': 5},
            {'name': 'Cilantro', 'unit': 'bunch', 'price': 50, 'shelf_life': 5},
            {'name': 'Parsley', 'unit': 'bunch', 'price': 50, 'shelf_life': 7},
            {'name': 'Black Pepper', 'unit': 'kg', 'price': 800, 'shelf_life': 365},
            {'name': 'Oregano', 'unit': 'kg', 'price': 650, 'shelf_life': 180},
            {'name': 'Cumin', 'unit': 'kg', 'price': 720, 'shelf_life': 180},
        ],
        'Prepared Foods': [
            {'name': 'Pasta Sauce', 'unit': 'jar', 'price': 200, 'shelf_life': 14},
            {'name': 'Chicken Stock', 'unit': 'liter', 'price': 160, 'shelf_life': 14},
            {'name': 'Hummus', 'unit': 'container', 'price': 180, 'shelf_life': 7},
            {'name': 'Salsa', 'unit': 'jar', 'price': 200, 'shelf_life': 14},
            {'name': 'Soup Base', 'unit': 'liter', 'price': 240, 'shelf_life': 21},
        ]
    }
    
    # Generate random inventory with varied quantities and expiry dates in 2025
    inventory = []
    # Set today to March 30, 2025 (as specified in the prompt)
    today = datetime(2025, 3, 30)
    
    for category, items in categories.items():
        # Add a random selection of items from each category
        # Manually create items with specific expiry categories to ensure we have all types
        # This ensures we have a good distribution of items in each expiry category
        expiry_categories = [
            (-10, -1),  # Expired
            (0, 3),     # Critical
            (4, 7),     # Warning
            (8, 14),    # Approaching
            (15, 30)    # Safe
        ]
        
        # Track how many items we've added in each category
        category_counts = {i: 0 for i in range(len(expiry_categories))}
        target_count_per_category = 5  # Aim for at least this many in each category
        
        # Shuffle items to get different ones in each category
        random.shuffle(items)
        
        for item in items:
            # Determine which expiry category to use for this item
            # Start by looking for categories that need more items
            possible_categories = [i for i, count in category_counts.items() 
                                if count < target_count_per_category]
            
            # If all categories have enough, pick one randomly
            if not possible_categories:
                expiry_group = random.randint(0, len(expiry_categories) - 1)
            else:
                expiry_group = random.choice(possible_categories)
                
            # Update count for this category
            category_counts[expiry_group] += 1
            
            # Set quantity and expiry days based on category
            quantity = random.randint(2, 15)
            days_range = expiry_categories[expiry_group]
            days_until_expiry = random.randint(days_range[0], days_range[1])
            
            # Calculate expiry date based on days until expiry
            expiry_date = today + timedelta(days=days_until_expiry)
            # Calculate received date working backwards from expiry
            received_date = expiry_date - timedelta(days=item['shelf_life'])
            
            inventory.append({
                'category': category,
                'item': item['name'],
                'quantity': quantity,
                'unit': item['unit'],
                'price': item['price'],
                'value': round(quantity * item['price'], 2),
                'received_date': received_date.strftime('%Y-%m-%d'),
                'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                'days_until_expiry': days_until_expiry
            })
    
    return inventory

def generate_sample_sales():
    """
    Generate sample sales data for demonstration purposes.
    
    Returns:
        DataFrame containing sample sales data
    """
    # Define popular dishes and their properties with prices in Indian Rupees (₹)
    dishes = [
        {'name': 'Grilled Chicken Salad', 'category': 'Salads', 'price': 399, 'popularity': 0.8},
        {'name': 'Pasta Primavera', 'category': 'Pasta', 'price': 450, 'popularity': 0.7},
        {'name': 'Steak Frites', 'category': 'Mains', 'price': 699, 'popularity': 0.9},
        {'name': 'Vegetable Curry', 'category': 'Mains', 'price': 499, 'popularity': 0.6},
        {'name': 'Fish Tacos', 'category': 'Mains', 'price': 550, 'popularity': 0.75},
        {'name': 'Caesar Salad', 'category': 'Salads', 'price': 350, 'popularity': 0.85},
        {'name': 'Spaghetti Bolognese', 'category': 'Pasta', 'price': 450, 'popularity': 0.8},
        {'name': 'Margherita Pizza', 'category': 'Pizza', 'price': 399, 'popularity': 0.9},
        {'name': 'Chocolate Cake', 'category': 'Desserts', 'price': 250, 'popularity': 0.7},
        {'name': 'Cheesecake', 'category': 'Desserts', 'price': 280, 'popularity': 0.8},
        {'name': 'Tiramisu', 'category': 'Desserts', 'price': 250, 'popularity': 0.6},
        {'name': 'French Onion Soup', 'category': 'Starters', 'price': 280, 'popularity': 0.5},
        {'name': 'Bruschetta', 'category': 'Starters', 'price': 250, 'popularity': 0.65},
        {'name': 'Caprese Salad', 'category': 'Starters', 'price': 299, 'popularity': 0.7}
    ]
    
    # Generate sales data for 90 days in 2025
    sales_data = []
    end_date = datetime(2025, 3, 30)  # Today (March 30, 2025)
    start_date = end_date - timedelta(days=90)
    current_date = start_date
    
    # Define weekly patterns (0=Monday, 6=Sunday)
    day_factors = {
        0: 0.7,  # Monday
        1: 0.8,  # Tuesday
        2: 0.9,  # Wednesday
        3: 1.0,  # Thursday
        4: 1.3,  # Friday
        5: 1.5,  # Saturday
        6: 1.2   # Sunday
    }
    
    # Generate sales records
    while current_date <= end_date:
        # Get day of week factor
        day_factor = day_factors[current_date.weekday()]
        
        # Number of orders for this day
        base_orders = 50  # Base number of orders
        daily_orders = int(base_orders * day_factor * random.uniform(0.8, 1.2))
        
        # Generate individual dish sales
        for _ in range(daily_orders):
            # Randomly select dishes with probability weighted by popularity
            for dish in dishes:
                # Probability of this dish being ordered
                if random.random() < dish['popularity'] * day_factor * 0.3:  # Adjust scaling as needed
                    quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])
                    
                    sales_data.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'dish': dish['name'],
                        'category': dish['category'],
                        'price': dish['price'],
                        'quantity': quantity,
                        'total': round(dish['price'] * quantity, 2)
                    })
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Convert to DataFrame
    df = pd.DataFrame(sales_data)
    
    # Add aggregated data
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    
    return df

def save_inventory_data(inventory_data):
    """
    Save inventory data to a CSV file.
    
    Args:
        inventory_data: List of dictionaries containing inventory data
    """
    # Convert to DataFrame
    df = pd.DataFrame(inventory_data)
    
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/sample_inventory.csv', index=False)
    print("Inventory data saved to data/sample_inventory.csv")

def save_sales_data(sales_data):
    """
    Save sales data to a CSV file.
    
    Args:
        sales_data: DataFrame containing sales data
    """
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    sales_data.to_csv('data/sample_sales.csv', index=False)
    print("Sales data saved to data/sample_sales.csv")
