import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta
import random

# Add the current directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility modules
from utils.menu_optimization import generate_recipe_recommendations, calculate_dish_costs, generate_new_recipe
from utils.data_loader import load_inventory_data, load_sales_data

st.set_page_config(
    page_title="Menu Optimizer | Smart Kitchen",
    page_icon="👨‍🍳",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = load_inventory_data()
    
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = load_sales_data()
    
if 'recommended_recipes' not in st.session_state:
    st.session_state.recommended_recipes = generate_recipe_recommendations(load_inventory_data())

# Page title and introduction
st.title("👨‍🍳 Menu Optimization")
st.write("AI-powered recipe recommendations, cost optimization, and menu planning based on your inventory and sales data.")

# Create tabs for different menu optimization features
tab1, tab2, tab3, tab4 = st.tabs(["Smart Recommendations", "Cost Analysis", "Menu Engineering", "AI Recipe Creator"])

# Tab 1: Smart Recommendations
with tab1:
    st.header("Recipe Recommendations Based on Inventory")
    st.write("These recommendations help you use ingredients that will expire soon, reducing waste and maximizing profits.")
    
    # Add refresh button to get new recommendations
    if st.button("Refresh Recommendations", key="refresh_recommendations"):
        st.session_state.recommended_recipes = generate_recipe_recommendations(st.session_state.inventory_data)
        st.success("Recommendations refreshed based on current inventory!")
    
    # Display recipe recommendations
    recipe_recommendations = st.session_state.recommended_recipes
    
    if recipe_recommendations:
        # Create a multi-card layout for recipes
        cols = st.columns(len(recipe_recommendations))
        
        for i, recipe in enumerate(recipe_recommendations):
            with cols[i]:
                st.subheader(recipe['name'])
                st.markdown(f"**Cuisine:** {recipe.get('cuisine', 'Mixed')}")
                st.markdown("**Uses expiring items:**")
                expiring_items = recipe.get('ingredients', [])[:4]  # Show first 4 ingredients
                for item in expiring_items:
                    st.markdown(f"• {item}")
                
                if len(recipe.get('ingredients', [])) > 4:
                    st.markdown(f"• ...and {len(recipe.get('ingredients', [])) - 4} more")
                
                st.markdown(f"**Profit margin:** {recipe.get('profit_margin', 0):.0%}")
                st.markdown(f"**Potential savings:** ₹{recipe.get('potential_savings', 0):.2f}")
                st.markdown(f"**Expires soon items used:** {recipe.get('expires_soon_used', 0)}")
                
                # Add button to view full recipe and add to menu
                if st.button("View Full Recipe", key=f"view_recipe_{i}"):
                    st.session_state.selected_recipe = recipe
                    st.rerun()
                
                if st.button("Add to Menu", key=f"add_menu_{i}"):
                    st.success(f"'{recipe['name']}' added to your menu!")
    else:
        st.info("No recipe recommendations available. Try refreshing or update your inventory.")
    
    # Display selected recipe details if any
    if 'selected_recipe' in st.session_state:
        st.subheader(f"Recipe Details: {st.session_state.selected_recipe['name']}")
        
        recipe = st.session_state.selected_recipe
        
        # Create a two-column layout for recipe details
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Ingredients")
            for ingredient in recipe.get('ingredients', []):
                st.markdown(f"• {ingredient}")
            
            st.markdown("### Preparation")
            for i, step in enumerate(recipe.get('instructions', []), 1):
                st.markdown(f"{i}. {step}")
            
            st.markdown(f"**Difficulty:** {recipe.get('difficulty', 'Medium')}")
            st.markdown(f"**Prep Time:** {recipe.get('prep_time', 30)} minutes")
        
        with col2:
            st.markdown("### Financial Analysis")
            st.markdown(f"**Base Cost:** ₹{recipe.get('base_cost', 0):.2f}")
            st.markdown(f"**Selling Price:** ₹{recipe.get('selling_price', 0):.2f}")
            st.markdown(f"**Profit Margin:** {recipe.get('profit_margin', 0):.0%}")
            st.markdown(f"**Potential Savings:** ₹{recipe.get('potential_savings', 0):.2f}")
            
            st.markdown("### Expiry Utilization")
            st.markdown(f"**Expiring Soon Items Used:** {recipe.get('expires_soon_used', 0)}")
            
            st.markdown("### Popularity Prediction")
            st.markdown(f"**Predicted Popularity:** {recipe.get('popularity', 0.7):.0%}")
        
        # Add to menu button
        if st.button("Add to Menu", key="add_menu_detail"):
            st.success(f"'{recipe['name']}' added to your menu!")
        
        # Close button
        if st.button("Close Recipe Details"):
            del st.session_state.selected_recipe
            st.rerun()

# Tab 2: Cost Analysis
with tab2:
    st.header("Menu Cost Analysis")
    st.write("Analyze your menu items to optimize costs and maximize profits.")
    
    # Sample dishes for analysis with prices in Indian Rupees (₹)
    sample_dishes = [
        {"name": "Grilled Chicken Salad", "category": "Salads", "base_cost": 601.75, "selling_price": 1406.85, "ingredients": ["Chicken", "Lettuce", "Tomatoes", "Carrots", "Herbs (Basil)"]},
        {"name": "Vegetable Curry", "category": "Mains", "base_cost": 539.50, "selling_price": 1323.85, "ingredients": ["Bell Peppers", "Carrots", "Potatoes", "Onions"]},
        {"name": "Spaghetti Bolognese", "category": "Pasta", "base_cost": 477.25, "selling_price": 1240.85, "ingredients": ["Ground Beef", "Tomatoes", "Onions", "Herbs (Basil)"]},
        {"name": "Fish Tacos", "category": "Mains", "base_cost": 705.50, "selling_price": 1489.85, "ingredients": ["Salmon Fillets", "Lettuce", "Tomatoes", "Lemons"]},
        {"name": "Margherita Pizza", "category": "Pizza", "base_cost": 373.50, "selling_price": 1157.85, "ingredients": ["Tomatoes", "Herbs (Basil)", "Cheddar Cheese"]},
        {"name": "Tiramisu", "category": "Desserts", "base_cost": 352.75, "selling_price": 742.85, "ingredients": ["Eggs", "Heavy Cream", "Coffee"]}
    ]
    
    # Let user select a dish to analyze
    selected_dish_name = st.selectbox("Select a dish to analyze", [dish["name"] for dish in sample_dishes])
    selected_dish = next((dish for dish in sample_dishes if dish["name"] == selected_dish_name), None)
    
    if selected_dish:
        # Calculate cost breakdown
        cost_analysis = calculate_dish_costs(selected_dish, st.session_state.inventory_data)
        
        # Display cost breakdown
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Cost Breakdown")
            
            # Create pie chart for ingredient costs
            ingredient_costs = cost_analysis.get('ingredient_costs', {})
            ingredients = list(ingredient_costs.keys())
            costs = list(ingredient_costs.values())
            
            if ingredients and costs:
                fig = px.pie(
                    names=ingredients,
                    values=costs,
                    title="Ingredient Cost Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed cost breakdown
            st.markdown("### Detailed Cost Breakdown")
            
            # Ingredient costs table
            st.markdown("**Ingredient Costs:**")
            for ingredient, cost in ingredient_costs.items():
                st.markdown(f"• {ingredient}: ₹{cost:.2f}")
            
            # Other costs
            st.markdown(f"**Labor Cost:** ₹{cost_analysis.get('labor_cost', 0):.2f}")
            st.markdown(f"**Overhead Cost:** ₹{cost_analysis.get('overhead_cost', 0):.2f}")
            st.markdown(f"**Total Cost:** ₹{cost_analysis.get('total_cost', 0):.2f}")
        
        with col2:
            st.subheader("Profitability Analysis")
            
            # Create a gauge chart for profit margin
            profit_margin = cost_analysis.get('profit_margin', 0)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=profit_margin * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Profit Margin"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 50], 'color': "orange"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': profit_margin * 100
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Financial summary
            st.markdown("### Financial Summary")
            st.markdown(f"**Selling Price:** ₹{cost_analysis.get('selling_price', 0):.2f}")
            st.markdown(f"**Total Cost:** ₹{cost_analysis.get('total_cost', 0):.2f}")
            st.markdown(f"**Profit per Serving:** ₹{cost_analysis.get('profit', 0):.2f}")
            st.markdown(f"**Profit Margin:** {profit_margin:.1%}")
            
            # Recommendations based on analysis
            st.markdown("### Recommendations")
            
            if profit_margin < 0.3:
                st.error("Low profit margin! Consider increasing price or reducing costs.")
            elif profit_margin < 0.5:
                st.warning("Moderate profit margin. Review ingredient costs for optimization.")
            else:
                st.success("Healthy profit margin. This dish is financially optimized.")
            
            # Specific recommendations
            most_expensive = max(ingredient_costs.items(), key=lambda x: x[1]) if ingredient_costs else (None, 0)
            if most_expensive[0]:
                st.markdown(f"• {most_expensive[0]} is your most expensive ingredient (₹{most_expensive[1]:.2f}). Consider alternatives or optimized portions.")
            
            if cost_analysis.get('labor_cost', 0) > cost_analysis.get('total_cost', 0) * 0.3:
                st.markdown("• Labor cost is high for this dish. Consider streamlining preparation.")
            
            # Price optimization suggestion
            optimal_price = cost_analysis.get('total_cost', 0) / (1 - 0.65)  # Target 65% margin
            if optimal_price > cost_analysis.get('selling_price', 0) * 1.1:
                st.markdown(f"• Consider increasing price to ₹{optimal_price:.2f} for optimal profitability.")
    else:
        st.info("Please select a dish to analyze.")

# Tab 3: Menu Engineering
with tab3:
    st.header("Menu Engineering Analysis")
    st.write("Analyze your menu items based on popularity and profitability to optimize your menu.")
    
    # Create sample menu data (in a real app, this would come from the sales data)
    sales_data = st.session_state.sales_data
    
    # Process sales data for menu engineering
    if 'dish' in sales_data.columns and 'total' in sales_data.columns and 'quantity' in sales_data.columns:
        # Aggregate sales by dish
        menu_performance = sales_data.groupby('dish').agg({
            'total': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # Add profitability data (in a real app, this would come from cost analysis)
        menu_performance['cost'] = 0
        menu_performance['profit'] = 0
        menu_performance['profit_margin'] = 0
        
        # For each dish, assign estimated cost and calculate profit
        for i, row in menu_performance.iterrows():
            dish_name = row['dish']
            estimated_cost_per_unit = row['total'] / row['quantity'] * random.uniform(0.3, 0.6)  # 30-60% cost ratio
            menu_performance.at[i, 'cost'] = estimated_cost_per_unit * row['quantity']
            menu_performance.at[i, 'profit'] = row['total'] - menu_performance.at[i, 'cost']
            menu_performance.at[i, 'profit_margin'] = menu_performance.at[i, 'profit'] / row['total']
        
        # Calculate averages for quadrant analysis
        avg_profit = menu_performance['profit'].mean()
        avg_quantity = menu_performance['quantity'].mean()
        
        # Assign quadrants
        def assign_quadrant(row):
            if row['profit'] >= avg_profit and row['quantity'] >= avg_quantity:
                return "Stars"
            elif row['profit'] >= avg_profit and row['quantity'] < avg_quantity:
                return "Puzzles"
            elif row['profit'] < avg_profit and row['quantity'] >= avg_quantity:
                return "Plow Horses"
            else:
                return "Dogs"
        
        menu_performance['quadrant'] = menu_performance.apply(assign_quadrant, axis=1)
        
        # Display menu engineering quadrant chart
        st.subheader("Menu Engineering Quadrant Analysis")
        
        # Create quadrant chart
        fig = px.scatter(menu_performance, x='quantity', y='profit',
                        size='total', color='quadrant',
                        hover_name='dish',
                        labels={'quantity': 'Popularity (Sales Volume)', 'profit': 'Profitability (₹)'},
                        title="Menu Engineering Matrix",
                        color_discrete_map={
                            'Stars': 'green',
                            'Puzzles': 'blue',
                            'Plow Horses': 'orange',
                            'Dogs': 'red'
                        })
        
        # Add quadrant lines
        fig.add_hline(y=avg_profit, line_dash="dash", line_color="gray")
        fig.add_vline(x=avg_quantity, line_dash="dash", line_color="gray")
        
        # Add annotations for quadrants
        fig.add_annotation(
            x=menu_performance['quantity'].max() * 0.75,
            y=menu_performance['profit'].max() * 0.75,
            text="STARS",
            showarrow=False,
            font=dict(size=14, color="green")
        )
        
        fig.add_annotation(
            x=menu_performance['quantity'].max() * 0.75,
            y=menu_performance['profit'].min() * 0.75,
            text="PLOW HORSES",
            showarrow=False,
            font=dict(size=14, color="orange")
        )
        
        fig.add_annotation(
            x=menu_performance['quantity'].min() * 1.25,
            y=menu_performance['profit'].max() * 0.75,
            text="PUZZLES",
            showarrow=False,
            font=dict(size=14, color="blue")
        )
        
        fig.add_annotation(
            x=menu_performance['quantity'].min() * 1.25,
            y=menu_performance['profit'].min() * 0.75,
            text="DOGS",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display quadrant-specific recommendations
        st.subheader("Strategic Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Stars (High Profit, High Popularity)")
            stars = menu_performance[menu_performance['quadrant'] == 'Stars']
            if not stars.empty:
                for _, dish in stars.iterrows():
                    st.markdown(f"• **{dish['dish']}**: Feature prominently, maintain quality")
            else:
                st.info("No dishes in this quadrant")
            
            st.markdown("### Puzzles (High Profit, Low Popularity)")
            puzzles = menu_performance[menu_performance['quadrant'] == 'Puzzles']
            if not puzzles.empty:
                for _, dish in puzzles.iterrows():
                    st.markdown(f"• **{dish['dish']}**: Reposition, promote more, consider placement")
            else:
                st.info("No dishes in this quadrant")
        
        with col2:
            st.markdown("### Plow Horses (Low Profit, High Popularity)")
            plow_horses = menu_performance[menu_performance['quadrant'] == 'Plow Horses']
            if not plow_horses.empty:
                for _, dish in plow_horses.iterrows():
                    st.markdown(f"• **{dish['dish']}**: Increase price or reduce costs")
            else:
                st.info("No dishes in this quadrant")
            
            st.markdown("### Dogs (Low Profit, Low Popularity)")
            dogs = menu_performance[menu_performance['quadrant'] == 'Dogs']
            if not dogs.empty:
                for _, dish in dogs.iterrows():
                    st.markdown(f"• **{dish['dish']}**: Remove or completely revamp")
            else:
                st.info("No dishes in this quadrant")
        
        # Display detailed menu performance table
        st.subheader("Detailed Menu Performance")
        
        # Format the table for display
        display_df = menu_performance.copy()
        display_df['profit_margin'] = display_df['profit_margin'].apply(lambda x: f"{x:.1%}")
        display_df['total'] = display_df['total'].apply(lambda x: f"₹{x:.2f}")
        display_df['cost'] = display_df['cost'].apply(lambda x: f"₹{x:.2f}")
        display_df['profit'] = display_df['profit'].apply(lambda x: f"₹{x:.2f}")
        
        st.dataframe(display_df)
    else:
        st.info("No comprehensive sales data available for menu engineering analysis.")

# Tab 4: AI Recipe Creator
with tab4:
    st.header("AI Recipe Creator")
    st.write("Generate new recipes based on ingredients you need to use up.")
    
    # Get list of ingredients from inventory
    inventory_df = pd.DataFrame(st.session_state.inventory_data)
    
    if 'item' in inventory_df.columns and 'expiry_date' in inventory_df.columns:
        # Convert expiry date to datetime if it's not already
        if not pd.api.types.is_datetime64_dtype(inventory_df['expiry_date']):
            inventory_df['expiry_date'] = pd.to_datetime(inventory_df['expiry_date'])
        
        # Find soon-to-expire ingredients
        today = pd.Timestamp.now()
        inventory_df['days_until_expiry'] = (inventory_df['expiry_date'] - today).dt.days
        
        # Sort inventory by expiry date
        inventory_df = inventory_df.sort_values('days_until_expiry')
        
        # Display items that are expiring soon
        soon_expiring = inventory_df[inventory_df['days_until_expiry'] <= 7]
        
        if not soon_expiring.empty:
            st.subheader("Ingredients Expiring Soon")
            
            expiring_items = soon_expiring['item'].tolist()
            
            # Create a multiselect for ingredient selection
            selected_ingredients = st.multiselect(
                "Select ingredients to use in your recipe",
                options=expiring_items,
                default=expiring_items[:min(4, len(expiring_items))]
            )
            
            # Define cuisine options
            cuisines = ["Italian", "Mediterranean", "Asian Fusion", "American", "Mexican", "Any"]
            
            # Create a selectbox for cuisine selection
            selected_cuisine = st.selectbox("Select cuisine type", cuisines)
            
            if selected_cuisine == "Any":
                selected_cuisine = None
            
            if selected_ingredients:
                # Generate button
                if st.button("Generate Recipe", key="generate_recipe"):
                    with st.spinner("Generating AI recipe..."):
                        # Generate a recipe using the selected ingredients
                        generated_recipe = generate_new_recipe(selected_ingredients, selected_cuisine)
                        st.session_state.generated_recipe = generated_recipe
                
                # Display generated recipe if available
                if 'generated_recipe' in st.session_state:
                    recipe = st.session_state.generated_recipe
                    
                    st.success("Recipe successfully generated!")
                    
                    st.subheader(f"🍽️ {recipe['name']}")
                    st.markdown(f"**Cuisine:** {recipe['cuisine']}")
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown("### Ingredients")
                        for ingredient in recipe['ingredients']:
                            st.markdown(f"• {ingredient}")
                        
                        st.markdown("### Instructions")
                        for i, instruction in enumerate(recipe['instructions'], 1):
                            st.markdown(f"{i}. {instruction}")
                    
                    with col2:
                        st.markdown("### Details")
                        st.markdown(f"**Cooking Method:** {recipe['cooking_method']}")
                        st.markdown(f"**Estimated Cost:** ₹{recipe['estimated_cost']:.2f}")
                        st.markdown(f"**Suggested Price:** ₹{recipe['estimated_price']:.2f}")
                        st.markdown(f"**Profit Margin:** {(recipe['estimated_price'] - recipe['estimated_cost']) / recipe['estimated_price']:.1%}")
                        st.markdown(f"**Expiring Ingredients Used:** {recipe['expiring_ingredients_used']}")
                        
                        # Add to menu button
                        if st.button("Add to Menu", key="add_generated_recipe"):
                            st.success(f"'{recipe['name']}' added to your menu!")
                        
                        # Create new recipe button
                        if st.button("Generate Another Recipe"):
                            del st.session_state.generated_recipe
                            st.rerun()
            else:
                st.info("Please select at least one ingredient to generate a recipe.")
        else:
            st.info("No ingredients are expiring within the next week.")
    else:
        st.error("Inventory data is not properly formatted or is missing expiry dates.")

# Footer
st.markdown("---")
st.markdown("## Next Steps")
st.write("Looking to further optimize your kitchen operations? Check out these other tools:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📦 [Inventory Tracking](/inventory_tracking)")
    st.write("Monitor your inventory levels and expiry dates")

with col2:
    st.markdown("### 📈 [Demand Prediction](/demand_prediction)")
    st.write("Forecast demand and optimize ordering")

with col3:
    st.markdown("### ♻️ [Waste Analysis](/waste_analysis)")
    st.write("Track and reduce food waste")
