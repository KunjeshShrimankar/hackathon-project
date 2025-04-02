import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility modules
from utils.waste_analysis import analyze_waste, calculate_waste_metrics, generate_waste_heatmap_data
from utils.data_loader import load_inventory_data, load_sales_data

st.set_page_config(
    page_title="Waste Analysis | Smart Kitchen",
    page_icon="♻️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = load_inventory_data()
    
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = load_sales_data()

# Generate or load waste data (in a real app, this would be loaded from a database)
if 'waste_data' not in st.session_state:
    # Sample waste data for demonstration with values in Indian Rupees (₹) using 2025 dates
    sample_waste_data = [
        {"date": "2025-03-20", "category": "Vegetables", "item": "Lettuce", "quantity_kg": 1.2, "value": 497.17, "reason": "Expired"},              # Converted from $5.99
        {"date": "2025-03-20", "category": "Vegetables", "item": "Tomatoes", "quantity_kg": 0.8, "value": 264.77, "reason": "Quality issues"},      # Converted from $3.19
        {"date": "2025-03-21", "category": "Protein", "item": "Chicken", "quantity_kg": 0.5, "value": 539.50, "reason": "Over-production"},         # Converted from $6.50
        {"date": "2025-03-22", "category": "Dairy", "item": "Milk", "quantity_kg": 1.0, "value": 314.57, "reason": "Expired"},                      # Converted from $3.79
        {"date": "2025-03-23", "category": "Fruits", "item": "Strawberries", "quantity_kg": 0.3, "value": 372.67, "reason": "Quality issues"},      # Converted from $4.49
        {"date": "2025-03-24", "category": "Vegetables", "item": "Bell Peppers", "quantity_kg": 0.4, "value": 165.17, "reason": "Trim waste"},      # Converted from $1.99
        {"date": "2025-03-24", "category": "Protein", "item": "Beef", "quantity_kg": 0.7, "value": 1446.69, "reason": "Over-production"},           # Converted from $17.43
        {"date": "2025-03-25", "category": "Dairy", "item": "Cheese", "quantity_kg": 0.2, "value": 232.40, "reason": "Expired"},                    # Converted from $2.80
        {"date": "2025-03-25", "category": "Vegetables", "item": "Carrots", "quantity_kg": 0.6, "value": 113.71, "reason": "Trim waste"},           # Converted from $1.37
        {"date": "2025-03-26", "category": "Fruits", "item": "Bananas", "quantity_kg": 0.9, "value": 133.63, "reason": "Quality issues"},           # Converted from $1.61
        {"date": "2025-03-27", "category": "Prepared Foods", "item": "Pasta Sauce", "quantity_kg": 0.5, "value": 207.50, "reason": "Over-production"}, # Converted from $2.50
        {"date": "2025-03-28", "category": "Vegetables", "item": "Onions", "quantity_kg": 0.3, "value": 49.80, "reason": "Trim waste"},             # Converted from $0.60
        {"date": "2025-03-29", "category": "Protein", "item": "Fish", "quantity_kg": 0.4, "value": 826.68, "reason": "Quality issues"}              # Converted from $9.96
    ]
    
    st.session_state.waste_data = sample_waste_data

# Page title and introduction
st.title("♻️ Waste Tracking & Analysis")
st.write("Track, analyze, and reduce food waste in your kitchen with AI-powered insights.")

# Create tabs for different waste analysis views
tab1, tab2, tab3, tab4 = st.tabs(["Waste Dashboard", "Waste Tracking", "Waste Analysis", "Waste Reduction"])

# Tab 1: Waste Dashboard
with tab1:
    st.header("Waste Analytics Dashboard")
    
    # Analyze waste data
    waste_analysis = analyze_waste(st.session_state.waste_data)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Waste (kg)", 
            value=f"{waste_analysis['total_waste_kg']:.1f} kg", 
            delta="-2.3 kg"
        )
    
    with col2:
        st.metric(
            label="Total Waste Value", 
            value=f"₹{waste_analysis['total_waste_value']:.2f}", 
            delta="-₹18.42"
        )
    
    with col3:
        # Calculate waste as percentage of inventory value
        inventory_value = sum(item.get('value', 0) for item in st.session_state.inventory_data)
        waste_percentage = (waste_analysis['total_waste_value'] / inventory_value * 100) if inventory_value > 0 else 0
        
        st.metric(
            label="Waste % of Inventory", 
            value=f"{waste_percentage:.1f}%", 
            delta="-1.2%"
        )
    
    with col4:
        # Calculate waste cost per customer (using sales data)
        total_customers = len(st.session_state.sales_data) / 2.5  # Assuming average party size of 2.5
        waste_per_customer = waste_analysis['total_waste_value'] / total_customers if total_customers > 0 else 0
        
        st.metric(
            label="Waste Cost per Customer", 
            value=f"₹{waste_per_customer:.2f}", 
            delta="-₹0.15"
        )
    
    # Display waste by category chart
    st.subheader("Waste by Category")
    
    waste_by_category = waste_analysis['waste_by_category']
    category_data = []
    
    for category, data in waste_by_category.items():
        category_data.append({
            'category': category,
            'quantity_kg': data['quantity_kg'],
            'value': data['value'],
            'percentage': data['percentage']
        })
    
    category_df = pd.DataFrame(category_data)
    
    # Create a horizontal bar chart for waste by category
    fig_category = px.bar(
        category_df,
        y='category',
        x='quantity_kg',
        color='value',
        orientation='h',
        labels={'quantity_kg': 'Waste (kg)', 'category': 'Category', 'value': 'Value (₹)'},
        title="Waste by Category",
        color_continuous_scale=px.colors.sequential.Reds
    )
    
    st.plotly_chart(fig_category, use_container_width=True)
    
    # Display waste by reason chart
    st.subheader("Waste by Reason")
    
    waste_by_reason = waste_analysis['waste_by_reason']
    reason_data = []
    
    for reason, data in waste_by_reason.items():
        reason_data.append({
            'reason': reason,
            'quantity_kg': data['quantity_kg'],
            'value': data['value'],
            'percentage': data['percentage']
        })
    
    reason_df = pd.DataFrame(reason_data)
    
    # Create a pie chart for waste by reason
    fig_reason = px.pie(
        reason_df,
        values='quantity_kg',
        names='reason',
        title="Waste Distribution by Reason",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    st.plotly_chart(fig_reason, use_container_width=True)
    
    # Display waste trend over time
    st.subheader("Waste Trend")
    
    trend_data = waste_analysis['trend_data']
    trend_df = pd.DataFrame({
        'date': list(trend_data.keys()),
        'waste_kg': list(trend_data.values())
    })
    
    trend_df['date'] = pd.to_datetime(trend_df['date'])
    trend_df = trend_df.sort_values('date')
    
    # Create a line chart for waste trend
    fig_trend = px.line(
        trend_df,
        x='date',
        y='waste_kg',
        labels={'date': 'Date', 'waste_kg': 'Waste (kg)'},
        title="Daily Waste Trend",
        line_shape='spline'
    )
    
    fig_trend.update_traces(line=dict(width=3))
    
    # Add a trend line
    fig_trend.add_trace(
        go.Scatter(
            x=trend_df['date'],
            y=trend_df['waste_kg'].rolling(window=7).mean(),
            mode='lines',
            name='7-Day Average',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Display insights and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Insights")
        for insight in waste_analysis['insights']:
            st.markdown(f"• {insight}")
    
    with col2:
        st.subheader("Recommendations")
        for recommendation in waste_analysis['recommendations']:
            st.markdown(f"• {recommendation}")

# Tab 2: Waste Tracking
with tab2:
    st.header("Waste Tracking & Logging")
    
    # Create a form to log new waste
    st.subheader("Log New Waste")
    
    with st.form("waste_log_form"):
        # Form fields
        col1, col2 = st.columns(2)
        
        with col1:
            waste_date = st.date_input("Date", datetime(2025, 3, 30))  # Use March 30, 2025 as the default date
            
            # Get categories from inventory data
            inventory_df = pd.DataFrame(st.session_state.inventory_data)
            categories = inventory_df['category'].unique() if 'category' in inventory_df.columns else [
                "Vegetables", "Fruits", "Protein", "Dairy", "Grains", "Herbs & Spices", "Prepared Foods"
            ]
            
            waste_category = st.selectbox("Category", categories)
            
            # Get items based on selected category
            items_in_category = inventory_df[inventory_df['category'] == waste_category]['item'].unique() if 'category' in inventory_df.columns else []
            waste_item = st.selectbox("Item", items_in_category if len(items_in_category) > 0 else ["Please select an item"])
            
            waste_quantity = st.number_input("Quantity (kg)", min_value=0.01, step=0.1, value=0.5)
        
        with col2:
            # Get item value from inventory if possible
            item_price = 0
            if waste_item in inventory_df['item'].values:
                item_data = inventory_df[inventory_df['item'] == waste_item]
                if 'price' in item_data.columns:
                    item_price = item_data['price'].iloc[0]
            
            waste_value = st.number_input("Value (₹)", min_value=0.0, step=0.1, value=item_price * waste_quantity if item_price > 0 else 0.0)
            
            waste_reasons = ["Expired", "Over-production", "Damaged", "Trim waste", "Quality issues", "Other"]
            waste_reason = st.selectbox("Reason", waste_reasons)
            
            waste_notes = st.text_area("Notes", "")
        
        # Submit button
        submit_button = st.form_submit_button("Log Waste")
        
        if submit_button:
            # In a real app, this would save to a database
            # Here we'll add to the session state list
            new_waste_entry = {
                "date": waste_date.strftime("%Y-%m-%d"),
                "category": waste_category,
                "item": waste_item,
                "quantity_kg": waste_quantity,
                "value": waste_value,
                "reason": waste_reason,
                "notes": waste_notes
            }
            
            st.session_state.waste_data.append(new_waste_entry)
            st.success(f"Logged {waste_quantity}kg of {waste_item} waste!")
    
    # Display waste log
    st.subheader("Waste Log")
    
    # Convert waste data to DataFrame for display
    waste_df = pd.DataFrame(st.session_state.waste_data)
    
    if not waste_df.empty and 'date' in waste_df.columns:
        # Add date filtering
        waste_df['date'] = pd.to_datetime(waste_df['date'])
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", waste_df['date'].min())
        with col2:
            end_date = st.date_input("End Date", datetime(2025, 3, 30))  # Use March 30, 2025 as the default date
        
        # Filter by date range
        filtered_waste = waste_df[
            (waste_df['date'] >= pd.Timestamp(start_date)) & 
            (waste_df['date'] <= pd.Timestamp(end_date))
        ]
        
        # Additional filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = ["All"] + sorted(waste_df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories)
        
        with col2:
            reasons = ["All"] + sorted(waste_df['reason'].unique().tolist())
            selected_reason = st.selectbox("Filter by Reason", reasons)
        
        with col3:
            # Sort options
            sort_options = {
                "Date (newest first)": ("date", False),
                "Date (oldest first)": ("date", True),
                "Value (highest first)": ("value", False),
                "Value (lowest first)": ("value", True),
                "Quantity (highest first)": ("quantity_kg", False),
                "Quantity (lowest first)": ("quantity_kg", True),
            }
            
            selected_sort = st.selectbox("Sort by", list(sort_options.keys()))
        
        # Apply category filter if selected
        if selected_category != "All":
            filtered_waste = filtered_waste[filtered_waste['category'] == selected_category]
        
        # Apply reason filter if selected
        if selected_reason != "All":
            filtered_waste = filtered_waste[filtered_waste['reason'] == selected_reason]
        
        # Apply sorting
        sort_column, ascending = sort_options[selected_sort]
        filtered_waste = filtered_waste.sort_values(by=sort_column, ascending=ascending)
        
        # Display the filtered waste log
        if not filtered_waste.empty:
            # Format for display
            display_waste = filtered_waste.copy()
            display_waste['date'] = display_waste['date'].dt.strftime('%Y-%m-%d')
            display_waste['value'] = display_waste['value'].apply(lambda x: f"₹{x:.2f}")
            
            # Reorder columns for better display
            display_columns = ['date', 'category', 'item', 'quantity_kg', 'value', 'reason']
            if 'notes' in display_waste.columns:
                display_columns.append('notes')
            
            st.dataframe(display_waste[display_columns], use_container_width=True)
            
            # Display summary statistics
            total_waste_kg = filtered_waste['quantity_kg'].sum()
            total_waste_value = filtered_waste['value'].sum()
            
            st.markdown(f"**Total:** {total_waste_kg:.1f}kg of waste worth ₹{total_waste_value:.2f}")
        else:
            st.info("No waste records match the selected filters.")
    else:
        st.info("No waste records available. Start logging waste using the form above.")

# Tab 3: Waste Analysis
with tab3:
    st.header("Detailed Waste Analysis")
    
    # Generate waste heatmap data
    heatmap_data = generate_waste_heatmap_data()
    
    # Display waste heatmap
    st.subheader("Kitchen Waste Heatmap")
    st.write("This visualization shows areas of your kitchen where waste is most prevalent.")
    
    # Convert the heatmap data to DataFrame for visualization
    heatmap_df = pd.DataFrame({
        'area': list(heatmap_data.keys()),
        'waste_level': list(heatmap_data.values())
    })
    
    # Sort by waste level for better visualization
    heatmap_df = heatmap_df.sort_values('waste_level', ascending=False)
    
    # Create a horizontal bar chart for the heatmap
    fig_heatmap = px.bar(
        heatmap_df,
        y='area',
        x='waste_level',
        orientation='h',
        labels={'waste_level': 'Waste Level', 'area': 'Kitchen Area'},
        title="Waste Heatmap by Kitchen Area",
        color='waste_level',
        color_continuous_scale=px.colors.sequential.Reds
    )
    
    fig_heatmap.update_layout(xaxis_range=[0, 1])
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Identify high waste areas
    high_waste_areas = heatmap_df[heatmap_df['waste_level'] > 0.6]['area'].tolist()
    
    if high_waste_areas:
        st.markdown("### High Waste Areas")
        st.markdown("The following areas have high waste levels and should be prioritized for improvement:")
        
        for area in high_waste_areas:
            st.markdown(f"• **{area}**: {heatmap_df[heatmap_df['area'] == area]['waste_level'].iloc[0]:.0%} waste level")
        
        st.markdown("### Recommendations for High Waste Areas")
        st.markdown("""
        **Prep Station:**
        - Implement standardized cutting techniques to reduce trim waste
        - Use clear portion control tools and guides
        - Train staff on efficient vegetable preparation
        
        **Salad Station:**
        - Store greens properly with moisture control
        - Implement just-in-time prep for leafy vegetables
        - Rotate stock religiously (FIFO - First In, First Out)
        
        **Vegetable Cooler:**
        - Adjust temperature and humidity settings
        - Improve organization with clear labeling and dating
        - Implement a color-coded system for expiry tracking
        """)
    
    # Waste analysis by time of day
    st.subheader("Waste Analysis by Time of Day")
    
    # Create sample data for time of day analysis
    time_periods = ["Morning Prep (6-10am)", "Lunch Service (11am-2pm)", 
                   "Afternoon Prep (2-5pm)", "Dinner Service (5-10pm)", "Closing (10pm-12am)"]
    
    # Converting USD to INR with factor of 83
    time_waste_data = [
        {"period": "Morning Prep (6-10am)", "waste_kg": 2.3, "waste_value": 18.75 * 83, "main_reason": "Trim waste"},
        {"period": "Lunch Service (11am-2pm)", "waste_kg": 1.8, "waste_value": 27.50 * 83, "main_reason": "Over-production"},
        {"period": "Afternoon Prep (2-5pm)", "waste_kg": 1.2, "waste_value": 14.25 * 83, "main_reason": "Trim waste"},
        {"period": "Dinner Service (5-10pm)", "waste_kg": 3.5, "waste_value": 52.80 * 83, "main_reason": "Over-production"},
        {"period": "Closing (10pm-12am)", "waste_kg": 4.7, "waste_value": 76.30 * 83, "main_reason": "Unsold food"}
    ]
    
    time_waste_df = pd.DataFrame(time_waste_data)
    
    # Create a bar chart for waste by time of day
    fig_time = px.bar(
        time_waste_df,
        x='period',
        y='waste_kg',
        color='waste_value',
        labels={'period': 'Time Period', 'waste_kg': 'Waste (kg)', 'waste_value': 'Value (₹)'},
        title="Waste by Time of Day",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Display waste by time period insights
    st.markdown("### Insights by Time Period")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Highest Waste Periods:**")
        closing_value = 76.30 * 83
        dinner_value = 52.80 * 83
        st.markdown(f"""
        1. **Closing (10pm-12am)**: 4.7kg waste, ₹{closing_value:.2f} value
        - **Primary Issue**: Unsold food at end of service
        - **Recommendation**: Implement dynamic batch cooking during last service hours
        
        2. **Dinner Service (5-10pm)**: 3.5kg waste, ₹{dinner_value:.2f} value
        - **Primary Issue**: Over-production during peak hours
        - **Recommendation**: Refine forecasting for dinner service variability
        """)
    
    with col2:
        st.markdown("**Opportunity Areas:**")
        st.markdown("""
        1. **Morning Prep (6-10am)**: 2.3kg waste
        - **Primary Issue**: Trim waste from vegetable preparation
        - **Recommendation**: Implement a trim usage program for stocks and sauces
        
        2. **Lunch Service (11am-2pm)**: 1.8kg waste
        - **Primary Issue**: Over-production for lunch rush
        - **Recommendation**: Refine lunch forecasting by day of week
        """)
    
    # Financial impact analysis
    st.subheader("Financial Impact Analysis")
    
    # Calculate waste as percentage of revenue
    waste_df = pd.DataFrame(st.session_state.waste_data)
    total_waste_value = waste_df['value'].sum() if 'value' in waste_df.columns else 0
    
    sales_df = st.session_state.sales_data
    total_revenue = sales_df['total'].sum() if 'total' in sales_df.columns else 0
    
    waste_percentage_of_revenue = (total_waste_value / total_revenue * 100) if total_revenue > 0 else 0
    industry_benchmark = 4.0  # Industry benchmark waste percentage
    
    # Create a gauge chart for waste percentage of revenue
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=waste_percentage_of_revenue,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Waste as % of Revenue"},
        delta={'reference': industry_benchmark, 'increasing': {'color': 'red'}, 'decreasing': {'color': 'green'}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 2], 'color': "green"},
                {'range': [2, 4], 'color': "yellow"},
                {'range': [4, 6], 'color': "orange"},
                {'range': [6, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': industry_benchmark
            }
        }
    ))
    
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Display financial impact
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Financial Impact")
        st.markdown(f"**Waste Value (Annual Projection):** ₹{total_waste_value * 365 / len(waste_df) if len(waste_df) > 0 else 0:,.2f}")
        st.markdown(f"**Waste as % of Revenue:** {waste_percentage_of_revenue:.1f}%")
        st.markdown(f"**Industry Benchmark:** {industry_benchmark:.1f}%")
        
        performance_statement = "below" if waste_percentage_of_revenue < industry_benchmark else "above"
        st.markdown(f"Your waste percentage is **{performance_statement}** the industry benchmark.")
    
    with col2:
        st.markdown("### Potential Savings")
        
        # Calculate potential savings
        target_percentage = min(waste_percentage_of_revenue, industry_benchmark) * 0.75
        potential_savings_percentage = waste_percentage_of_revenue - target_percentage
        annual_savings = total_revenue * potential_savings_percentage / 100
        
        st.markdown(f"**Target Waste Percentage:** {target_percentage:.1f}%")
        st.markdown(f"**Potential Annual Savings:** ₹{annual_savings:,.2f}")
        st.markdown(f"**Impact on Profit Margin:** +{potential_savings_percentage:.1f}%")
        
        # Calculate ROI of waste reduction
        st.markdown(f"**ROI of Waste Reduction Program:** 300-500%")

# Tab 4: Waste Reduction
with tab4:
    st.header("Waste Reduction Strategies")
    
    # Create waste reduction plan
    st.subheader("Waste Reduction Action Plan")
    
    # Generate waste analysis for insights
    waste_analysis = analyze_waste(st.session_state.waste_data)
    
    # Identify top waste categories
    waste_by_category = waste_analysis['waste_by_category']
    top_waste_categories = sorted(waste_by_category.items(), key=lambda x: x[1]['value'], reverse=True)
    top_waste_categories = top_waste_categories[:3] if len(top_waste_categories) >= 3 else top_waste_categories
    
    # Identify top waste reasons
    waste_by_reason = waste_analysis['waste_by_reason']
    top_waste_reasons = sorted(waste_by_reason.items(), key=lambda x: x[1]['value'], reverse=True)
    top_waste_reasons = top_waste_reasons[:3] if len(top_waste_reasons) >= 3 else top_waste_reasons
    
    # Display action plan
    st.markdown("Based on your waste data, we've created a targeted action plan to reduce waste in your kitchen:")
    
    # Top category strategies
    st.markdown("### Priority Categories")
    
    for category, data in top_waste_categories:
        st.markdown(f"#### 1. {category} ({data['quantity_kg']:.1f}kg, ₹{data['value']:.2f})")
        
        if category == "Vegetables":
            st.markdown("""
            **Strategies:**
            - Implement just-in-time preparation for leafy greens and highly perishable items
            - Store vegetables at optimal temperature and humidity levels
            - Train staff on efficient cutting techniques to reduce trim waste
            - Create a "vegetable scrap" program for stocks, soups, and sauces
            - Track usage patterns by day of week to optimize ordering
            """)
        elif category == "Protein":
            st.markdown("""
            **Strategies:**
            - Implement portion control tools and standardized recipes
            - Train staff on proper butchering techniques to maximize yield
            - Use vacuum sealing to extend shelf life of expensive proteins
            - Cross-utilize proteins across multiple menu items
            - Explore sous-vide cooking for better yield and longer shelf life
            """)
        elif category == "Dairy":
            st.markdown("""
            **Strategies:**
            - Implement strict FIFO (First In, First Out) rotation protocols
            - Store dairy products at the optimal temperature (34-38°F)
            - Use opened items in staff meals before they expire
            - Break down bulk cheese in-house to prevent surface mold
            - Monitor expiration dates daily
            """)
        elif category == "Fruits":
            st.markdown("""
            **Strategies:**
            - Order fruits at varying ripeness levels to ensure continuous supply
            - Repurpose overripe fruits for desserts, sauces, and preserves
            - Store fruits at the proper temperature and humidity level
            - Train staff on proper cutting techniques to maximize yield
            - Consider frozen options for off-season items
            """)
        elif category == "Prepared Foods":
            st.markdown("""
            **Strategies:**
            - Implement batch cooking during service hours
            - Use smaller serving vessels towards the end of service
            - Properly cool and store leftovers for next-day specials
            - Create a daily "special" utilizing prepared foods from the previous day
            - Train staff on portion control and demand forecasting
            """)
        else:
            st.markdown("""
            **Strategies:**
            - Implement better inventory tracking and rotation
            - Train staff on proper storage and handling
            - Optimize ordering based on usage patterns
            - Cross-utilize ingredients across multiple menu items
            - Repurpose excess before it becomes waste
            """)
    
    # Top reason strategies
    st.markdown("### Priority Causes")
    
    for reason, data in top_waste_reasons:
        st.markdown(f"#### 1. {reason} ({data['quantity_kg']:.1f}kg, ₹{data['value']:.2f})")
        
        if reason == "Expired":
            st.markdown("""
            **Strategies:**
            - Implement digital inventory tracking with expiration alerts
            - Train all staff on FIFO (First In, First Out) protocols
            - Use color-coded labels for quick expiry identification
            - Conduct daily inventory checks of high-risk items
            - Adjust order quantities and frequency for perishable items
            """)
        elif reason == "Over-production":
            st.markdown("""
            **Strategies:**
            - Implement data-driven forecasting based on historical sales
            - Use batch cooking throughout service periods
            - Train kitchen staff on scaling recipes accurately
            - Create a flexible menu with cross-utilization of ingredients
            - Establish end-of-night protocols for handling leftovers
            """)
        elif reason == "Trim waste":
            st.markdown("""
            **Strategies:**
            - Train staff on efficient cutting techniques
            - Implement standardized cutting procedures with visual guides
            - Collect trim waste separately for stocks, sauces, and soups
            - Analyze trim waste to identify training opportunities
            - Consider purchasing pre-cut items for difficult products
            """)
        elif reason == "Quality issues":
            st.markdown("""
            **Strategies:**
            - Improve receiving protocols and inspection procedures
            - Build better relationships with suppliers and provide feedback
            - Adjust storage conditions to maintain quality longer
            - Train staff on proper handling to prevent bruising and damage
            - Implement quality checks throughout storage period
            """)
        elif reason == "Damaged":
            st.markdown("""
            **Strategies:**
            - Train staff on proper storage techniques
            - Improve organization in walk-ins and dry storage
            - Implement safe transportation methods between storage areas
            - Use proper containers designed for food storage
            - Conduct regular staff training on handling procedures
            """)
        else:
            st.markdown("""
            **Strategies:**
            - Analyze the specific causes and implement targeted solutions
            - Train staff on best practices for handling and storage
            - Improve processes to prevent recurring issues
            - Document issues to identify patterns over time
            - Schedule regular waste reduction meetings with key staff
            """)
    
    # Staff training program
    st.subheader("Staff Training Program")
    
    st.markdown("""
    ### Waste Reduction Training Modules
    
    1. **Inventory Management & FIFO (30 min)**
       - Proper receiving procedures
       - FIFO principles and practice
       - Labeling and rotation systems
       - Daily inventory checks
    
    2. **Cutting Techniques & Yield Optimization (45 min)**
       - High-yield cutting methods
       - Maximizing usable product
       - Trim collection for secondary use
       - Portion control techniques
    
    3. **Forecasting & Production Planning (30 min)**
       - Using historical data to predict needs
       - Batch cooking techniques
       - Scaling recipes up and down accurately
       - Communication between FOH and BOH
    
    4. **Storage & Preservation Techniques (30 min)**
       - Optimal storage conditions by food type
       - Vacuum sealing and other preservation methods
       - Signs of spoilage and quality issues
       - Extending shelf life safely
    
    5. **Waste Tracking & Analysis (15 min)**
       - Using the waste tracking system
       - Understanding waste metrics
       - Personal responsibility and accountability
       - Setting and achieving waste reduction goals
    """)
    
    # Waste tracking goals
    st.subheader("Waste Reduction Goals")
    
    # Calculate current waste metrics
    waste_df = pd.DataFrame(st.session_state.waste_data)
    total_waste_kg = waste_df['quantity_kg'].sum() if 'quantity_kg' in waste_df.columns else 0
    total_waste_value = waste_df['value'].sum() if 'value' in waste_df.columns else 0
    
    # Set realistic reduction goals
    short_term_goal = total_waste_kg * 0.85  # 15% reduction
    medium_term_goal = total_waste_kg * 0.70  # 30% reduction
    long_term_goal = total_waste_kg * 0.50  # 50% reduction
    
    # Create a goals tracker
    goals_data = {
        'Timeline': ['Current', '30 Days (Short-term)', '90 Days (Medium-term)', '180 Days (Long-term)'],
        'Target (kg)': [total_waste_kg, short_term_goal, medium_term_goal, long_term_goal],
        'Target Value (₹)': [total_waste_value, total_waste_value * 0.85, total_waste_value * 0.70, total_waste_value * 0.50],
        'Status': ['Baseline', 'In Progress', 'Planned', 'Planned']
    }
    
    goals_df = pd.DataFrame(goals_data)
    
    # Create a line chart for waste reduction goals
    fig_goals = px.line(
        goals_df,
        x='Timeline',
        y='Target (kg)',
        markers=True,
        labels={'Timeline': 'Goal Timeline', 'Target (kg)': 'Waste Target (kg)'},
        title="Waste Reduction Goals",
    )
    
    fig_goals.update_traces(line=dict(width=3))
    
    # Add value as text on the chart
    for i, row in goals_df.iterrows():
        fig_goals.add_annotation(
            x=row['Timeline'],
            y=row['Target (kg)'],
            text=f"₹{row['Target Value (₹)']:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30
        )
    
    st.plotly_chart(fig_goals, use_container_width=True)
    
    # Display goal tracking
    st.markdown("""
    ### Goal Tracking Methodology
    
    1. **Daily Waste Logging**
       - All kitchen staff record waste in the tracking system
       - Weigh and categorize all waste before disposal
       - Note the reason for waste and estimated value
    
    2. **Weekly Review Meetings**
       - Review waste trends from the previous week
       - Identify problem areas and success stories
       - Adjust strategies based on findings
       - Recognize staff who contribute to waste reduction
    
    3. **Monthly Goal Assessment**
       - Compare current waste metrics to goals
       - Celebrate successes and identify challenges
       - Adjust goals or strategies as needed
       - Share financial impact with team
    
    4. **Reward System**
       - Set up a bonus pool tied to waste reduction
       - Distribute monthly rewards for achieving targets
       - Recognize individual contributions to waste reduction
       - Create friendly competition between shifts or stations
    """)
    
    # Circular economy initiatives
    st.subheader("Circular Economy Initiatives")
    
    st.markdown("""
    ### Beyond Waste Reduction: Creating a Circular Kitchen
    
    1. **Composting Program**
       - Partner with local composting services
       - Turn unavoidable food waste into valuable compost
       - Reduce landfill impact and greenhouse gas emissions
    
    2. **Oil Recycling**
       - Recycle used cooking oil into biodiesel
       - Reduce disposal costs and environmental impact
       - Partner with local biodiesel producers
    
    3. **Packaging Reduction**
       - Work with suppliers to reduce packaging waste
       - Use reusable containers where possible
       - Select recyclable or compostable packaging
    
    4. **Food Donation Program**
       - Partner with local food banks or shelters
       - Donate safe, unused food to those in need
       - Reduce waste while supporting the community
    
    5. **Farm Partnerships**
       - Return organic waste to local farms for animal feed or composting
       - Create a closed-loop system with suppliers
       - Support local sustainable agriculture
    """)
    
    # ROI calculator for waste reduction initiatives
    st.subheader("Waste Reduction ROI Calculator")
    
    st.write("Calculate the return on investment for implementing waste reduction initiatives:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        annual_food_cost = st.number_input("Annual Food Cost (₹)", min_value=0, value=500000 * 83)
        current_waste_percentage = st.slider("Current Waste Percentage (%)", min_value=1, max_value=20, value=8)
        target_waste_percentage = st.slider("Target Waste Percentage (%)", min_value=1, max_value=current_waste_percentage, value=4)
        implementation_cost = st.number_input("Implementation Cost (₹)", min_value=0, value=5000 * 83)
    
    with col2:
        # Calculate results
        current_waste_value = annual_food_cost * (current_waste_percentage / 100)
        target_waste_value = annual_food_cost * (target_waste_percentage / 100)
        annual_savings = current_waste_value - target_waste_value
        roi_percentage = (annual_savings / implementation_cost) * 100 if implementation_cost > 0 else 0
        payback_months = (implementation_cost / annual_savings) * 12 if annual_savings > 0 else 0
        
        st.metric("Current Annual Waste Value", f"₹{current_waste_value:,.2f}")
        st.metric("Target Annual Waste Value", f"₹{target_waste_value:,.2f}")
        st.metric("Annual Savings", f"₹{annual_savings:,.2f}")
        st.metric("ROI", f"{roi_percentage:.1f}%")
        st.metric("Payback Period", f"{payback_months:.1f} months")
    
    # Create a pie chart comparing current and target waste
    waste_comparison = pd.DataFrame({
        'Status': ['Current Waste', 'Target Waste', 'Savings'],
        'Value': [current_waste_value, target_waste_value, annual_savings]
    })
    
    fig_roi = px.pie(
        waste_comparison,
        values='Value',
        names='Status',
        title="Waste Reduction Impact",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    st.plotly_chart(fig_roi, use_container_width=True)
    
    if roi_percentage > 200:
        st.success("This waste reduction initiative has an excellent ROI and short payback period!")
    elif roi_percentage > 100:
        st.success("This waste reduction initiative has a good ROI and reasonable payback period.")
    else:
        st.info("This waste reduction initiative has a positive ROI but consider adjusting your targets or implementation costs.")

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
    st.markdown("### 👨‍🍳 [Menu Optimizer](/menu_optimizer)")
    st.write("Create recipes to minimize waste")
