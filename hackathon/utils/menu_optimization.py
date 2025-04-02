import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_recipe_recommendations(inventory_data):
    """
    Generate recipe recommendations based on current inventory,
    focusing on items that will expire soon.
    
    Args:
        inventory_data: List of inventory items with expiry dates
        
    Returns:
        List of recommended recipes that utilize soon-to-expire ingredients
    """
    # In a real implementation, we would:
    # 1. Identify soon-to-expire ingredients
    # 2. Search a recipe database for dishes that use these ingredients
    # 3. Rank recipes based on ingredient usage, popularity, and profitability
    
    # For this demo, we'll return simulated recipe recommendations
    
    # Define a set of template recipes that could be recommended with prices in Indian Rupees (₹)
    recipe_templates = [
        {
            "name": "Herb-Crusted Chicken with Roasted Vegetables",
            "ingredients": ["Chicken", "Herbs (Basil)", "Bell Peppers", "Potatoes", "Onions"],
            "base_cost": 705.50,  # Converted from $8.50
            "selling_price": 1572.85,  # Converted from $18.95
            "profit_margin": 0.55,
            "difficulty": "Medium",
            "prep_time": 25,
            "popularity": 0.85
        },
        {
            "name": "Fresh Garden Vegetable Soup",
            "ingredients": ["Tomatoes", "Carrots", "Onions", "Herbs (Basil)", "Potatoes"],
            "base_cost": 352.75,  # Converted from $4.25
            "selling_price": 991.85,  # Converted from $11.95
            "profit_margin": 0.64,
            "difficulty": "Easy",
            "prep_time": 30,
            "popularity": 0.72
        },
        {
            "name": "Citrus Glazed Salmon",
            "ingredients": ["Salmon", "Lemons", "Herbs (Basil)", "Bell Peppers"],
            "base_cost": 809.25,  # Converted from $9.75
            "selling_price": 1904.85,  # Converted from $22.95
            "profit_margin": 0.58,
            "difficulty": "Medium",
            "prep_time": 20,
            "popularity": 0.88
        },
        {
            "name": "Classic Caesar Salad with Grilled Chicken",
            "ingredients": ["Chicken", "Lettuce", "Lemons", "Herbs (Basil)"],
            "base_cost": 456.50,  # Converted from $5.50
            "selling_price": 1240.85,  # Converted from $14.95
            "profit_margin": 0.63,
            "difficulty": "Easy",
            "prep_time": 15,
            "popularity": 0.75
        },
        {
            "name": "Ratatouille",
            "ingredients": ["Tomatoes", "Bell Peppers", "Onions", "Herbs (Basil)"],
            "base_cost": 518.75,  # Converted from $6.25
            "selling_price": 1406.85,  # Converted from $16.95
            "profit_margin": 0.63,
            "difficulty": "Medium",
            "prep_time": 35,
            "popularity": 0.70
        },
        {
            "name": "Vegetable Curry with Rice",
            "ingredients": ["Bell Peppers", "Carrots", "Potatoes", "Onions"],
            "base_cost": 477.25,  # Converted from $5.75
            "selling_price": 1323.85,  # Converted from $15.95
            "profit_margin": 0.64,
            "difficulty": "Medium",
            "prep_time": 30,
            "popularity": 0.82
        }
    ]
    
    # For a real implementation, we would match these recipes against actual inventory
    # Here, we'll randomly select a few and adjust their attributes to simulate the recommendation process
    
    # Select 3 recipes randomly for demonstration
    recommended_recipes = random.sample(recipe_templates, min(3, len(recipe_templates)))
    
    # Add some randomness to the profitability and other metrics
    for recipe in recommended_recipes:
        # Slightly adjust profit margin to simulate freshness impact
        recipe['profit_margin'] = min(0.75, max(0.5, recipe['profit_margin'] + random.uniform(-0.05, 0.05)))
        
        # Add a description for the UI
        recipe['description'] = f"A delicious {recipe['name'].lower()} that makes the most of your inventory."
        
        # Add savings information (in Indian Rupees)
        recipe['potential_savings'] = round(random.uniform(664, 2075), 2)  # Converted from $8-$25 to ₹
        
        # Add expiry utilization
        recipe['expires_soon_used'] = random.randint(2, 4)
    
    return recommended_recipes

def calculate_dish_costs(recipe, inventory_data):
    """
    Calculate the precise cost of a dish based on current inventory costs.
    
    Args:
        recipe: Recipe dictionary with ingredients
        inventory_data: Current inventory with cost information
        
    Returns:
        Dictionary with cost breakdown and profitability metrics
    """
    # In a real implementation, we would:
    # 1. Match recipe ingredients to inventory items
    # 2. Calculate exact costs based on required quantities
    # 3. Compute profit metrics
    
    # For this demo, we'll return simulated cost calculations
    
    # Start with base cost from recipe (in Indian Rupees)
    base_cost = recipe.get('base_cost', 622.50)  # Default 7.50 USD converted to INR
    
    # Add some random variation to simulate actual cost calculation
    actual_cost = base_cost * random.uniform(0.9, 1.1)
    
    # Calculate other financial metrics
    selling_price = recipe.get('selling_price', 1489.85)  # Default 17.95 USD converted to INR
    profit = selling_price - actual_cost
    profit_margin = profit / selling_price
    
    # Create a breakdown of costs
    ingredient_costs = {}
    for ingredient in recipe.get('ingredients', []):
        ingredient_costs[ingredient] = round(actual_cost * random.uniform(0.1, 0.3), 2)
    
    # Ensure the total matches the actual cost
    adjustment_factor = actual_cost / sum(ingredient_costs.values())
    ingredient_costs = {k: round(v * adjustment_factor, 2) for k, v in ingredient_costs.items()}
    
    # Create the cost analysis result
    cost_analysis = {
        'recipe_name': recipe.get('name', 'Recipe'),
        'total_cost': round(actual_cost, 2),
        'selling_price': selling_price,
        'profit': round(profit, 2),
        'profit_margin': round(profit_margin, 4),
        'ingredient_costs': ingredient_costs,
        'labor_cost': round(actual_cost * 0.3, 2),  # Estimated labor cost
        'overhead_cost': round(actual_cost * 0.15, 2)  # Estimated overhead
    }
    
    return cost_analysis

def generate_new_recipe(expiring_ingredients, cuisine_type=None):
    """
    Use AI to generate a new recipe based on expiring ingredients.
    
    Args:
        expiring_ingredients: List of ingredients that are about to expire
        cuisine_type: Optional cuisine type to focus on
        
    Returns:
        Dictionary with the new recipe details
    """
    # In a real implementation, we would:
    # 1. Use a generative AI model like GPT to create a new recipe
    # 2. Ensure it maximizes the use of soon-to-expire ingredients
    # 3. Format it nicely for presentation
    
    # For this demo, we'll return a simulated AI-generated recipe
    
    cuisines = ["Italian", "Mediterranean", "Asian Fusion", "American", "Mexican"]
    selected_cuisine = cuisine_type if cuisine_type else random.choice(cuisines)
    
    # Generate a recipe name based on ingredients and cuisine
    primary_ingredient = random.choice(expiring_ingredients)
    
    # Define cooking method first for all cuisines
    cooking_method = ""
    
    if selected_cuisine == "Italian":
        cooking_method = random.choice(["sautéed", "roasted", "pan-seared"])
        recipe_name = f"{primary_ingredient} Pasta with {expiring_ingredients[1] if len(expiring_ingredients) > 1 else 'Herb'} Sauce"
    elif selected_cuisine == "Mediterranean":
        cooking_method = random.choice(["grilled", "baked", "braised"])
        recipe_name = f"{cooking_method} {primary_ingredient} with {expiring_ingredients[1] if len(expiring_ingredients) > 1 else 'Lemon'} and Herbs"
    elif selected_cuisine == "Asian Fusion":
        recipe_name = f"{primary_ingredient} Stir-Fry with {expiring_ingredients[1] if len(expiring_ingredients) > 1 else 'Ginger'}"
        cooking_method = random.choice(["stir-fried", "steamed", "wok-tossed"])
    elif selected_cuisine == "American":
        recipe_name = f"{primary_ingredient} Burger with {expiring_ingredients[1] if len(expiring_ingredients) > 1 else 'Special'} Sauce"
        cooking_method = random.choice(["grilled", "pan-fried", "smoked"])
    else:  # Mexican
        recipe_name = f"{primary_ingredient} Tacos with {expiring_ingredients[1] if len(expiring_ingredients) > 1 else 'Lime'} Salsa"
        cooking_method = random.choice(["grilled", "slow-cooked", "roasted"])
    
    # Create the recipe structure
    recipe = {
        "name": recipe_name,
        "cuisine": selected_cuisine,
        "ingredients": expiring_ingredients + ["Salt", "Pepper", "Olive oil"],
        "cooking_method": cooking_method,
        "instructions": [
            f"Prepare the {primary_ingredient} by washing and cutting into appropriate pieces.",
            f"Season with salt, pepper, and any additional spices.",
            f"{cooking_method.capitalize()} until properly cooked.",
            "Combine with other ingredients according to taste.",
            "Serve hot with appropriate garnish."
        ],
        "estimated_cost": round(random.uniform(415, 996), 2),  # Converted from $5-$12 to ₹
        "estimated_price": round(random.uniform(1162, 2075), 2),  # Converted from $14-$25 to ₹
        "ai_generated": True,
        "expiring_ingredients_used": len(expiring_ingredients)
    }
    
    return recipe
