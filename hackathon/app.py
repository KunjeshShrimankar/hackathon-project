import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io

# Add the current directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.computer_vision import detect_objects_in_image, detect_food_spoilage
from utils.forecasting import forecast_demand, predict_waste
from utils.menu_optimization import generate_recipe_recommendations
from utils.waste_analysis import analyze_waste, calculate_waste_metrics
from utils.data_loader import load_inventory_data, load_sales_data

# Page configuration
st.set_page_config(
    page_title="Smart Kitchen & Waste Minimizer",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'inventory_data' not in st.session_state:
    # Force regeneration to ensure we have 2025 dates
    st.session_state.inventory_data = load_inventory_data(force_regenerate=True)
    
if 'sales_data' not in st.session_state:
    # Force regeneration to ensure we have 2025 dates
    st.session_state.sales_data = load_sales_data(force_regenerate=True)

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
    
if 'detected_items' not in st.session_state:
    st.session_state.detected_items = []
    
if 'spoilage_results' not in st.session_state:
    st.session_state.spoilage_results = []

# Application header and description
st.title("🍳 AI-Powered Smart Kitchen & Waste Minimizer")
st.markdown("""
This platform helps restaurant owners optimize kitchen operations, reduce waste, and improve profitability 
through computer vision, machine learning, and advanced analytics.
""")

# Main dashboard metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Items in Inventory", 
        value=len(st.session_state.inventory_data), 
        delta="4 new items"
    )
    
with col2:
    waste_percentage = 12.5
    st.metric(
        label="Food Waste %", 
        value=f"{waste_percentage}%", 
        delta="-2.5%"
    )
    
with col3:
    revenue_impact = calculate_waste_metrics(st.session_state.inventory_data)
    st.metric(
        label="Potential Revenue Impact", 
        value=f"₹{revenue_impact:,.2f}", 
        delta="15%"
    )
    
with col4:
    # Use March 30, 2025 as the reference date instead of system date
    reference_date = pd.Timestamp('2025-03-30')
    expiry_count = sum(1 for item in st.session_state.inventory_data 
                      if pd.to_datetime(item['expiry_date']) < reference_date + pd.Timedelta(days=3))
    st.metric(
        label="Items Expiring Soon", 
        value=expiry_count, 
        delta_color="inverse",
        delta="+2"
    )

# Main dashboard layout
st.markdown("## 🔍 Quick Actions")

# Computer vision section
vision_col1, vision_col2 = st.columns(2)

with vision_col1:
    st.subheader("Visual Inventory Scanner")
    uploaded_file = st.file_uploader("Upload an image of your inventory to analyze", 
                                    type=['jpg', 'jpeg', 'png'])
    
    use_webcam = st.checkbox("Use webcam instead")
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file.read()
        image = Image.open(io.BytesIO(st.session_state.uploaded_image))
        st.image(image, caption="Uploaded Inventory Image", use_column_width=True)
        
        if st.button("Analyze Inventory"):
            with st.spinner("Analyzing inventory with computer vision..."):
                # Process the image and detect objects
                detected_items = detect_objects_in_image(st.session_state.uploaded_image)
                st.session_state.detected_items = detected_items
                
                # Check for spoilage
                spoilage_results = detect_food_spoilage(st.session_state.uploaded_image)
                st.session_state.spoilage_results = spoilage_results
            
            st.success(f"Analysis complete! Detected {len(detected_items)} items.")
    
    elif use_webcam:
        st.write("Webcam functionality would be implemented here in a production environment.")
        # Note: Webcam functionality is limited in deployed Streamlit apps
        st.info("For security reasons, webcam access may be limited in deployed applications.")

with vision_col2:
    st.subheader("Detection Results")
    
    if hasattr(st.session_state, 'detected_items') and st.session_state.detected_items:
        # Display detected items
        st.write("📋 Detected Items:")
        
        detected_df = pd.DataFrame(st.session_state.detected_items)
        st.dataframe(detected_df)
        
        # Display spoilage alerts
        if st.session_state.spoilage_results:
            st.error("⚠️ Potential Food Spoilage Detected:")
            for item in st.session_state.spoilage_results:
                st.write(f"- {item['item']} ({item['confidence']:.1%} confidence)")
    else:
        st.info("Upload and analyze an image to see detection results here.")

# Forecasting section
st.markdown("## 📊 Key Insights")
forecast_col1, forecast_col2 = st.columns(2)

with forecast_col1:
    st.subheader("Demand Forecast")
    
    # Generate demand forecast
    forecast_data = forecast_demand(st.session_state.sales_data)
    
    # Display forecast chart
    st.line_chart(forecast_data)
    
    st.markdown("### Recommendations")
    st.info("Based on forecast, consider reducing inventory of salad greens by 15% and increasing protein items for next week.")

with forecast_col2:
    st.subheader("Waste Prediction")
    
    # Generate waste prediction
    waste_prediction = predict_waste(st.session_state.inventory_data, st.session_state.sales_data)
    
    # Display waste prediction
    st.bar_chart(waste_prediction)
    
    st.markdown("### High-Risk Items")
    high_risk_items = [
        {"item": "Fresh herbs", "risk_factor": 0.87, "days_to_expiry": 2},
        {"item": "Tomatoes", "risk_factor": 0.76, "days_to_expiry": 3},
        {"item": "Seafood mix", "risk_factor": 0.92, "days_to_expiry": 1}
    ]
    
    for item in high_risk_items:
        st.warning(f"{item['item']}: {item['risk_factor']:.0%} risk, {item['days_to_expiry']} days until expiry")

# Recipe recommendations section
st.markdown("## 👨‍🍳 Menu Optimization")

# Generate recommendations based on inventory
recipe_recommendations = generate_recipe_recommendations(st.session_state.inventory_data)

rec1, rec2, rec3 = st.columns(3)

with rec1:
    if len(recipe_recommendations) > 0:
        recipe = recipe_recommendations[0]
        st.markdown(f"### {recipe['name']}")
        st.markdown(f"**Uses:** {', '.join(recipe['ingredients'][:3])}...")
        st.markdown(f"**Profit margin:** {recipe['profit_margin']:.0%}")
        st.button("Add to menu", key="rec1")

with rec2:
    if len(recipe_recommendations) > 1:
        recipe = recipe_recommendations[1]
        st.markdown(f"### {recipe['name']}")
        st.markdown(f"**Uses:** {', '.join(recipe['ingredients'][:3])}...")
        st.markdown(f"**Profit margin:** {recipe['profit_margin']:.0%}")
        st.button("Add to menu", key="rec2")

with rec3:
    if len(recipe_recommendations) > 2:
        recipe = recipe_recommendations[2]
        st.markdown(f"### {recipe['name']}")
        st.markdown(f"**Uses:** {', '.join(recipe['ingredients'][:3])}...")
        st.markdown(f"**Profit margin:** {recipe['profit_margin']:.0%}")
        st.button("Add to menu", key="rec3")

# App footer with navigation to other pages
st.markdown("---")
st.markdown("## Explore More Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 📝 [Inventory Tracking](/inventory_tracking)")
    st.write("Detailed inventory management and tracking")

with col2:
    st.markdown("### 📈 [Demand Prediction](/demand_prediction)")
    st.write("AI-powered sales forecasting and analysis")

with col3:
    st.markdown("### 🍲 [Menu Optimizer](/menu_optimizer)")
    st.write("Smart recipe suggestions and menu planning")

with col4:
    st.markdown("### ♻️ [Waste Analysis](/waste_analysis)")
    st.write("Track and minimize food waste in your kitchen")

# Display system status
st.sidebar.title("System Status")
st.sidebar.success("All systems operational")
st.sidebar.metric("API Calls Remaining", "947", "Unlimited")
st.sidebar.metric("Storage Used", "42%", "-5%")

# Add data controls
st.sidebar.markdown("---")
st.sidebar.subheader("Data Controls")
if st.sidebar.button("Regenerate Sample Data"):
    # Regenerate sample inventory and sales data
    st.session_state.inventory_data = load_inventory_data(force_regenerate=True)
    st.session_state.sales_data = load_sales_data(force_regenerate=True)
    st.sidebar.success("Sample data regenerated!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.text("Smart Kitchen v1.0")
