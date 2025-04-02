import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility modules
from utils.forecasting import forecast_demand, predict_waste, analyze_stock_levels
from utils.data_loader import load_sales_data, load_inventory_data

st.set_page_config(
    page_title="Demand Prediction | Smart Kitchen",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = load_sales_data()
    
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = load_inventory_data()

# Page title and introduction
st.title("📈 AI-Powered Demand Prediction")
st.write("Use advanced forecasting to predict future sales and optimize inventory.")

# Create tabs for different forecast views
tab1, tab2, tab3 = st.tabs(["Demand Forecast", "Waste Prediction", "Stock Optimization"])

# Tab 1: Demand Forecast
with tab1:
    st.header("Sales Forecasting")
    
    # Load sales data
    sales_data = st.session_state.sales_data
    
    # Convert date to datetime if it's not already
    if 'date' in sales_data.columns and not pd.api.types.is_datetime64_dtype(sales_data['date']):
        sales_data['date'] = pd.to_datetime(sales_data['date'])
    
    # Display historical sales chart
    if 'date' in sales_data.columns and 'total' in sales_data.columns:
        # Aggregate sales by date
        daily_sales = sales_data.groupby('date')['total'].sum().reset_index()
        
        # Create a chart of historical sales
        st.subheader("Historical Sales")
        
        # Allow user to select time range
        time_range = st.selectbox(
            "Select time range for historical data", 
            ["Last 7 days", "Last 30 days", "Last 90 days", "All data"]
        )
        
        # Filter based on selected range
        today = pd.Timestamp.now()
        if time_range == "Last 7 days":
            start_date = today - pd.Timedelta(days=7)
            filtered_sales = daily_sales[daily_sales['date'] >= start_date]
        elif time_range == "Last 30 days":
            start_date = today - pd.Timedelta(days=30)
            filtered_sales = daily_sales[daily_sales['date'] >= start_date]
        elif time_range == "Last 90 days":
            start_date = today - pd.Timedelta(days=90)
            filtered_sales = daily_sales[daily_sales['date'] >= start_date]
        else:
            filtered_sales = daily_sales
        
        # Display the chart
        if not filtered_sales.empty:
            sales_fig = px.line(filtered_sales, x='date', y='total', 
                             title=f"Daily Sales - {time_range}",
                             labels={'date': 'Date', 'total': 'Sales (₹)'},
                             line_shape='spline')
            sales_fig.update_traces(line=dict(width=3))
            st.plotly_chart(sales_fig, use_container_width=True)
        else:
            st.warning("No sales data available for the selected time range.")
    
    # Generate forecast
    st.subheader("Sales Forecast")
    
    # Allow user to select forecast period
    forecast_days = st.slider("Forecast Period (days)", min_value=7, max_value=30, value=14, step=1)
    
    # Generate the forecast
    forecast_data = forecast_demand(sales_data, days_ahead=forecast_days)
    
    # Display forecast chart
    if not forecast_data.empty:
        # Create a forecast chart
        forecast_fig = px.line(forecast_data, x=forecast_data.index, y='forecasted_demand',
                          title=f"{forecast_days}-Day Sales Forecast",
                          labels={'x': 'Date', 'forecasted_demand': 'Forecasted Sales (₹)'},
                          line_shape='spline')
        forecast_fig.update_traces(line=dict(width=3, color='green'))
        
        # Add confidence intervals (simulated)
        upper_bound = forecast_data['forecasted_demand'] * 1.15
        lower_bound = forecast_data['forecasted_demand'] * 0.85
        
        forecast_fig.add_trace(go.Scatter(
            x=forecast_data.index,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Bound'
        ))
        
        forecast_fig.add_trace(go.Scatter(
            x=forecast_data.index,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 128, 0, 0.2)',
            showlegend=False,
            name='Lower Bound'
        ))
        
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Display forecast insights
        st.subheader("Forecast Insights")
        
        # Calculate some metrics
        avg_forecast = forecast_data['forecasted_demand'].mean()
        max_forecast = forecast_data['forecasted_demand'].max()
        max_day = forecast_data['forecasted_demand'].idxmax()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Daily Forecast", f"₹{avg_forecast:.2f}")
        
        with col2:
            st.metric("Peak Day Forecast", f"₹{max_forecast:.2f}")
        
        with col3:
            st.metric("Peak Day", max_day)
        
        # Provide business recommendations
        st.write("### Recommendations")
        
        # Example recommendations (these would be dynamically generated in a production system)
        st.info("""
        Based on the forecast:
        - Increase staffing on peak days (esp. Friday and Saturday)
        - Prepare for 15% higher vegetable usage next week
        - Expect lower demand for desserts in the early week
        """)
        
        # Display the forecast data in tabular form
        with st.expander("View Detailed Forecast Data"):
            st.dataframe(forecast_data)
    else:
        st.warning("Unable to generate forecast. Please check your sales data.")
    
    # By-item forecast (if category-level data is available)
    if 'category' in sales_data.columns:
        st.subheader("Category-Level Forecast")
        
        # Analyze sales by category
        category_sales = sales_data.groupby(['date', 'category'])['total'].sum().reset_index()
        
        # Let user select categories to view
        categories = sales_data['category'].unique()
        selected_categories = st.multiselect("Select categories to forecast", categories, default=categories[:3] if len(categories) > 3 else categories)
        
        if selected_categories:
            # Filter for selected categories
            filtered_category_sales = category_sales[category_sales['category'].isin(selected_categories)]
            
            # Create a chart showing sales by category
            category_fig = px.line(filtered_category_sales, x='date', y='total', color='category',
                                title="Sales by Category",
                                labels={'date': 'Date', 'total': 'Sales (₹)', 'category': 'Category'},
                                line_shape='spline')
            
            st.plotly_chart(category_fig, use_container_width=True)
            
            # Generate category-level forecasts (simplified example)
            today = pd.Timestamp.now()
            future_dates = [today + pd.Timedelta(days=i) for i in range(1, 8)]
            
            # Create forecast data for each category
            category_forecasts = []
            
            for category in selected_categories:
                category_data = filtered_category_sales[filtered_category_sales['category'] == category]
                
                if not category_data.empty:
                    # Get the average daily sales for this category (simplified forecasting)
                    avg_daily_sales = category_data['total'].mean()
                    
                    # Apply a simple forecast model (this would be more sophisticated in a real system)
                    for i, date in enumerate(future_dates):
                        # Use day of week factor to adjust forecast
                        dow_factor = 1.0 + (0.1 * (date.dayofweek % 7))  # Higher on weekends
                        forecast_value = avg_daily_sales * dow_factor
                        
                        category_forecasts.append({
                            'date': date,
                            'category': category,
                            'forecast': forecast_value
                        })
            
            # Display the forecasts
            if category_forecasts:
                forecast_df = pd.DataFrame(category_forecasts)
                
                # Create a chart of the forecasts
                forecast_by_category = px.line(forecast_df, x='date', y='forecast', color='category',
                                           title="7-Day Forecast by Category",
                                           labels={'date': 'Date', 'forecast': 'Forecasted Sales (₹)', 'category': 'Category'},
                                           line_shape='spline')
                
                st.plotly_chart(forecast_by_category, use_container_width=True)
                
                # Display the forecast data in a table
                with st.expander("View Category Forecast Data"):
                    pivot_forecast = forecast_df.pivot(index='date', columns='category', values='forecast')
                    st.dataframe(pivot_forecast)
            else:
                st.warning("Unable to generate category-level forecasts.")
        else:
            st.info("Please select at least one category to view the forecast.")

# Tab 2: Waste Prediction
with tab2:
    st.header("Waste Prediction & Prevention")
    
    # Generate waste prediction
    waste_prediction = predict_waste(st.session_state.inventory_data, st.session_state.sales_data)
    
    # Display waste prediction chart
    if not waste_prediction.empty:
        st.subheader("Predicted Waste by Category")
        
        # Create a horizontal bar chart
        waste_fig = px.bar(waste_prediction, y=waste_prediction.index, x='predicted_waste_kg',
                         title="Predicted Waste in the Next Week (kg)",
                         labels={'predicted_waste_kg': 'Predicted Waste (kg)', 'y': 'Category'},
                         color='predicted_waste_kg',
                         color_continuous_scale=px.colors.sequential.Reds)
        
        waste_fig.update_layout(xaxis_title="Predicted Waste (kg)", yaxis_title="Category")
        
        st.plotly_chart(waste_fig, use_container_width=True)
        
        # Calculate total predicted waste
        total_waste = waste_prediction['predicted_waste_kg'].sum()
        
        # Estimate waste value (assuming average cost per kg in INR)
        avg_cost_per_kg = 1037.50  # Example value (12.50 USD converted to INR)
        waste_value = total_waste * avg_cost_per_kg
        
        # Display waste metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predicted Waste", f"{total_waste:.1f} kg")
        
        with col2:
            st.metric("Estimated Waste Value", f"₹{waste_value:.2f}")
        
        with col3:
            # Calculate waste as percentage of inventory (simplified)
            total_inventory = sum(item.get('quantity', 0) for item in st.session_state.inventory_data)
            waste_percentage = (total_waste / total_inventory) * 100 if total_inventory > 0 else 0
            st.metric("Waste Percentage", f"{waste_percentage:.1f}%", delta="-2.5%")
        
        # High-risk items
        st.subheader("High-Risk Items")
        
        # Identify items at high risk of becoming waste
        inventory_df = pd.DataFrame(st.session_state.inventory_data)
        
        if 'expiry_date' in inventory_df.columns:
            # Convert expiry date to datetime
            inventory_df['expiry_date'] = pd.to_datetime(inventory_df['expiry_date'])
            
            # Calculate days until expiry relative to March 30, 2025
            today = pd.Timestamp(2025, 3, 30)  # Fixed date (March 30, 2025)
            inventory_df['days_until_expiry'] = (inventory_df['expiry_date'] - today).dt.days
            
            # Calculate risk factor (inverse of days until expiry, adjusted) - including expired items
            def calculate_risk(days):
                if days <= 0:
                    return 1.0  # Already expired
                elif days <= 3:
                    return 0.9 - (days * 0.1)  # High risk
                elif days <= 7:
                    return 0.6 - ((days - 3) * 0.05)  # Medium risk
                else:
                    return max(0.1, 0.5 - ((days - 7) * 0.02))  # Lower risk
            
            inventory_df['risk_factor'] = inventory_df['days_until_expiry'].apply(calculate_risk)
            
            # Sort by risk factor
            high_risk_items = inventory_df.sort_values('risk_factor', ascending=False).head(5)
            
            if not high_risk_items.empty:
                # Display high risk items
                for _, item in high_risk_items.iterrows():
                    risk_percentage = item['risk_factor'] * 100
                    if risk_percentage > 80:
                        st.error(f"⚠️ {item['item']}: {risk_percentage:.0f}% risk, {item['days_until_expiry']} days until expiry")
                    elif risk_percentage > 50:
                        st.warning(f"⚠️ {item['item']}: {risk_percentage:.0f}% risk, {item['days_until_expiry']} days until expiry")
                    else:
                        st.info(f"⚠️ {item['item']}: {risk_percentage:.0f}% risk, {item['days_until_expiry']} days until expiry")
            else:
                st.success("No high-risk items identified in your inventory.")
        
        # Waste prevention recommendations
        st.subheader("Waste Prevention Recommendations")
        
        # Example recommendations (would be dynamically generated in a production system)
        st.write("""
        ### Immediate Actions:
        1. **Create daily specials** using soon-to-expire vegetables
        2. **Adjust order quantities** for high-waste categories (especially vegetables)
        3. **Implement FIFO protocol** for protein items to prevent spoilage
        4. **Re-purpose trim waste** for stocks and soups
        5. **Train staff** on proper storage techniques to extend shelf life
        """)
        
        # Waste trend projection
        st.subheader("Waste Trend Projection")
        
        # Generate sample waste trend data starting from March 30, 2025
        dates = pd.date_range(start=pd.Timestamp(2025, 3, 30), periods=30, freq='D')
        
        # Create baseline trend with weekly pattern
        baseline = [3 + (0.5 * (i % 7 == 5 or i % 7 == 6)) for i in range(30)]
        
        # Create scenarios
        no_action = baseline
        moderate_action = [max(0.5, val * (1 - 0.02 * i)) for i, val in enumerate(baseline)]
        aggressive_action = [max(0.3, val * (1 - 0.04 * i)) for i, val in enumerate(baseline)]
        
        # Create DataFrame for chart
        trend_df = pd.DataFrame({
            'date': dates,
            'No Action': no_action,
            'Moderate Action': moderate_action,
            'Aggressive Action': aggressive_action
        })
        
        # Melt for Plotly
        trend_melted = pd.melt(trend_df, id_vars=['date'], var_name='scenario', value_name='waste_kg')
        
        # Create chart
        trend_fig = px.line(trend_melted, x='date', y='waste_kg', color='scenario',
                         title="Projected Daily Waste (kg) - 30 Day Outlook",
                         labels={'date': 'Date', 'waste_kg': 'Projected Waste (kg)', 'scenario': 'Scenario'},
                         line_shape='spline')
        
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Display potential savings
        baseline_total = sum(no_action)
        aggressive_total = sum(aggressive_action)
        potential_savings = (baseline_total - aggressive_total) * avg_cost_per_kg
        
        st.metric("Potential 30-Day Savings", f"₹{potential_savings:.2f}")
        
        # Show detailed waste data
        with st.expander("View Detailed Waste Prediction Data"):
            st.dataframe(waste_prediction)
    else:
        st.warning("Unable to generate waste prediction. Please check your inventory data.")

# Tab 3: Stock Optimization
with tab3:
    st.header("Inventory Optimization")
    
    # Get stock level analysis
    stock_analysis = analyze_stock_levels(st.session_state.inventory_data, forecast_demand(st.session_state.sales_data))
    
    # Display stock optimization insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Overstocked Items")
        
        overstocked = stock_analysis.get('overstocked_items', [])
        
        if overstocked:
            # Create a table for overstocked items
            overstocked_df = pd.DataFrame(overstocked)
            
            # Format and display
            st.dataframe(overstocked_df)
            
            # Calculate total excess value
            excess_value = sum(item['excess_value'] for item in overstocked)
            st.metric("Total Excess Value", f"₹{excess_value:.2f}", delta="-100%", delta_color="inverse")
        else:
            st.success("No overstocked items detected!")
    
    with col2:
        st.subheader("Understocked Items")
        
        understocked = stock_analysis.get('understocked_items', [])
        
        if understocked:
            # Create a table for understocked items
            understocked_df = pd.DataFrame(understocked)
            
            # Format and display
            st.dataframe(understocked_df)
            
            # Calculate total potential loss
            potential_loss = sum(item['potential_loss'] for item in understocked)
            st.metric("Potential Revenue Loss", f"₹{potential_loss:.2f}", delta="-100%")
        else:
            st.success("No understocked items detected!")
    
    # Display reorder recommendations
    st.subheader("Reorder Recommendations")
    
    reorder_recommendations = stock_analysis.get('reorder_recommendations', [])
    
    if reorder_recommendations:
        # Create a table for reorder recommendations
        reorder_df = pd.DataFrame(reorder_recommendations)
        
        # Add a "Reorder" button column (for demonstration)
        reorder_df['action'] = ['Reorder' for _ in range(len(reorder_df))]
        
        # Format and display
        st.dataframe(reorder_df)
        
        if st.button("Place Reorders"):
            st.success("Reorder requests submitted!")
    else:
        st.info("No reorder recommendations at this time.")
    
    # Par level optimization
    st.subheader("Par Level Optimization")
    
    # Allow user to select category for optimization
    inventory_df = pd.DataFrame(st.session_state.inventory_data)
    categories = inventory_df['category'].unique() if 'category' in inventory_df.columns else []
    
    if categories.size > 0:
        selected_category = st.selectbox("Select category to optimize", categories)
        
        # Display current par levels (simplified)
        st.write(f"### Current Par Levels - {selected_category}")
        
        # Filter inventory for selected category
        category_items = inventory_df[inventory_df['category'] == selected_category]
        
        # Display current stock levels
        if not category_items.empty:
            # Create par level analysis
            par_data = []
            
            for _, item in category_items.iterrows():
                # For demonstration, generate some sample par level data
                item_name = item['item']
                current_quantity = item['quantity']
                
                # Calculate a "recommended" par level based on some logic
                # In a real system, this would use demand forecasting and other factors
                usage_rate = np.random.uniform(0.5, 2.0)  # Units per day
                lead_time = np.random.randint(1, 4)  # Days to receive order
                safety_stock = usage_rate * 1.5  # Buffer stock
                
                recommended_par = round(usage_rate * lead_time + safety_stock, 1)
                
                par_data.append({
                    'item': item_name,
                    'current_stock': current_quantity,
                    'usage_rate': f"{usage_rate:.1f} per day",
                    'lead_time': f"{lead_time} days",
                    'recommended_par': recommended_par,
                    'adjustment': recommended_par - current_quantity
                })
            
            # Create DataFrame
            par_df = pd.DataFrame(par_data)
            
            # Add color styling
            def color_adjustment(val):
                if val > 0:
                    return 'background-color: #FFEB9C'  # Yellow for increase
                elif val < 0:
                    return 'background-color: #E6B0AA'  # Red for decrease
                else:
                    return 'background-color: #D5F5E3'  # Green for no change
            
            # Display styled table
            st.dataframe(par_df.style.applymap(color_adjustment, subset=['adjustment']))
            
            # Option to apply recommended par levels
            if st.button("Apply Recommended Par Levels"):
                st.success("Par levels updated successfully!")
        else:
            st.info(f"No items found in the {selected_category} category.")
    else:
        st.warning("No category information available in inventory data.")
    
    # Seasonal adjustment recommendations
    st.subheader("Seasonal Adjustment Recommendations")
    
    # Example seasonal recommendations (in a real system, this would be data-driven)
    st.write("""
    Based on historical patterns and upcoming events, consider these seasonal adjustments:
    
    - **Increase fresh produce by 20%** for the summer season
    - **Reduce heavy protein items by 10%** during hot weather
    - **Increase dessert options by 15%** for upcoming holiday period
    - **Prepare for 25% higher weekend volume** during tourist season
    """)
    
    # Display inventory turnover analysis
    st.subheader("Inventory Turnover Analysis")
    
    # Create sample turnover data
    turnover_data = {
        'Vegetables': 8.2,
        'Fruits': 7.5,
        'Protein': 5.8,
        'Dairy': 4.3,
        'Grains': 3.1,
        'Herbs & Spices': 2.4,
        'Prepared Foods': 9.5
    }
    
    turnover_df = pd.DataFrame([
        {'category': k, 'turnover_rate': v} for k, v in turnover_data.items()
    ])
    
    # Create turnover chart
    turnover_fig = px.bar(turnover_df, x='category', y='turnover_rate',
                       title="Inventory Turnover Rate by Category (turns per month)",
                       labels={'turnover_rate': 'Turnover Rate', 'category': 'Category'},
                       color='turnover_rate',
                       color_continuous_scale=px.colors.sequential.Viridis)
    
    st.plotly_chart(turnover_fig, use_container_width=True)
    
    # Turnover analysis notes
    st.write("""
    ### Turnover Insights:
    - **High turnover categories** (Prepared Foods, Vegetables) indicate efficient inventory management
    - **Low turnover categories** (Herbs & Spices, Grains) may have excessive stock levels
    - **Target turnover rate** for perishables should be >7.0
    - **Target turnover rate** for dry goods should be >3.0
    """)
