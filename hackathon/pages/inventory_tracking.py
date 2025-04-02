import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
from datetime import datetime, timedelta
import cv2
from PIL import Image
import io

# Add the current directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility modules
from utils.computer_vision import detect_objects_in_image, detect_food_spoilage
from utils.data_loader import load_inventory_data, save_inventory_data

st.set_page_config(
    page_title="Inventory Tracking | Smart Kitchen",
    page_icon="📦",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = load_inventory_data()
    
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
    
if 'detected_items' not in st.session_state:
    st.session_state.detected_items = []
    
if 'spoilage_results' not in st.session_state:
    st.session_state.spoilage_results = []

# Page title and introduction
st.title("📦 Smart Inventory Tracking")
st.write("Monitor your inventory in real-time, track expiry dates, and get smart restocking suggestions.")

# Create tabs for different inventory views
tab1, tab2, tab3, tab4 = st.tabs(["Inventory Dashboard", "Computer Vision Scanner", "Manage Inventory", "Expiry Tracking"])

# Tab 1: Inventory Dashboard
with tab1:
    st.header("Inventory Overview")
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    inventory_data = st.session_state.inventory_data
    
    # Convert to DataFrame for easier filtering
    inventory_df = pd.DataFrame(inventory_data)
    
    with col1:
        total_items = len(inventory_data)
        st.metric("Total Inventory Items", total_items)
    
    with col2:
        total_value = sum(item['value'] for item in inventory_data)
        st.metric("Total Inventory Value", f"₹{total_value:,.2f}")
    
    with col3:
        if 'expiry_date' in inventory_df.columns:
            inventory_df['expiry_date'] = pd.to_datetime(inventory_df['expiry_date'])
            expiring_soon = inventory_df[inventory_df['expiry_date'] < (pd.Timestamp.now() + pd.Timedelta(days=7))]
            expiring_count = len(expiring_soon)
            st.metric("Items Expiring in 7 Days", expiring_count, delta_color="inverse")
    
    with col4:
        low_stock_threshold = 3
        low_stock_count = len(inventory_df[inventory_df['quantity'] <= low_stock_threshold]) if 'quantity' in inventory_df.columns else 0
        st.metric("Items Low in Stock", low_stock_count, delta_color="inverse")
    
    # Display inventory by category chart
    st.subheader("Inventory by Category")
    
    if 'category' in inventory_df.columns:
        category_counts = inventory_df.groupby('category').size().reset_index(name='count')
        category_chart = px.pie(category_counts, values='count', names='category', 
                             title='Inventory Distribution by Category',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(category_chart, use_container_width=True)
    
    # Display inventory value by category
    st.subheader("Inventory Value by Category")
    
    if 'category' in inventory_df.columns and 'value' in inventory_df.columns:
        category_values = inventory_df.groupby('category')['value'].sum().reset_index()
        value_chart = px.bar(category_values, x='category', y='value', 
                          title='Inventory Value by Category',
                          labels={'value': 'Value (₹)', 'category': 'Category'},
                          color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(value_chart, use_container_width=True)
    
    # Display full inventory table
    st.subheader("Full Inventory")
    
    # Add search functionality
    search_term = st.text_input("Search Inventory", "")
    
    # Filter the inventory based on the search term
    if search_term:
        filtered_inventory = inventory_df[inventory_df.apply(lambda row: any(search_term.lower() in str(val).lower() for val in row), axis=1)]
    else:
        filtered_inventory = inventory_df
    
    # Display the filtered inventory
    st.dataframe(filtered_inventory, use_container_width=True)

# Tab 2: Computer Vision Scanner
with tab2:
    st.header("Visual Inventory Scanner")
    st.write("Upload images of your inventory to automatically detect and log items.")
    
    # Upload image for analysis
    uploaded_file = st.file_uploader("Upload an image of your inventory", type=['jpg', 'jpeg', 'png'])
    
    # Add option to use webcam (note: may not work in deployed Streamlit apps)
    use_webcam = st.checkbox("Use webcam instead")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            # Store the uploaded image in session state
            st.session_state.uploaded_image = uploaded_file.read()
            
            # Display the uploaded image
            image = Image.open(io.BytesIO(st.session_state.uploaded_image))
            st.image(image, caption="Uploaded Inventory Image", use_column_width=True)
            
            analyze_button = st.button("Analyze Inventory")
            
            if analyze_button:
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
            st.info("For security reasons, webcam access may be limited in deployed applications.")
    
    with col2:
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
            
            # Add detected items to inventory
            if st.button("Add Items to Inventory"):
                existing_inventory = st.session_state.inventory_data
                
                # Process and add each detected item
                for item in st.session_state.detected_items:
                    # Check if item already exists in inventory
                    existing_item = next((i for i in existing_inventory if i['item'] == item['item']), None)
                    
                    if existing_item:
                        # Update existing item quantity
                        existing_item['quantity'] += item['quantity']
                        existing_item['value'] = existing_item['quantity'] * existing_item.get('price', 0)
                    else:
                        # Create a new inventory item
                        today = datetime.now()
                        expiry_date = today + timedelta(days=item['expiry_days'])
                        
                        new_item = {
                            'category': 'Detected Items',  # Default category
                            'item': item['item'],
                            'quantity': item['quantity'],
                            'unit': 'unit',  # Default unit
                            'price': 0,  # Default price (to be updated)
                            'value': 0,  # To be calculated
                            'received_date': today.strftime('%Y-%m-%d'),
                            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                            'days_until_expiry': item['expiry_days']
                        }
                        
                        existing_inventory.append(new_item)
                
                # Update the session state
                st.session_state.inventory_data = existing_inventory
                
                # Save updated inventory
                save_inventory_data(existing_inventory)
                
                st.success("Items added to inventory!")
        else:
            st.info("Upload and analyze an image to see detection results here.")

# Tab 3: Manage Inventory
with tab3:
    st.header("Manage Inventory")
    
    # Create columns for adding new items and editing inventory
    add_col, edit_col = st.columns(2)
    
    with add_col:
        st.subheader("Add New Item")
        
        # Form for adding new items
        with st.form("add_item_form"):
            # Get input for each field
            item_name = st.text_input("Item Name")
            
            # Default categories
            categories = ["Vegetables", "Fruits", "Protein", "Dairy", "Grains", "Herbs & Spices", "Prepared Foods", "Other"]
            category = st.selectbox("Category", categories)
            
            quantity = st.number_input("Quantity", min_value=0.0, step=0.1)
            
            units = ["kg", "g", "liter", "ml", "unit", "piece", "dozen", "package", "container", "other"]
            unit = st.selectbox("Unit", units)
            
            price = st.number_input("Price per Unit (₹)", min_value=0.0, step=0.01)
            
            received_date = st.date_input("Received Date", datetime.now())
            shelf_life = st.number_input("Shelf Life (days)", min_value=1, step=1)
            
            # Calculate expiry date based on received date and shelf life
            expiry_date = received_date + timedelta(days=shelf_life)
            
            # Display the calculated expiry date
            st.write(f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')}")
            
            # Calculate value
            value = quantity * price
            
            # Submit button
            submit_button = st.form_submit_button("Add Item")
            
            if submit_button:
                if not item_name:
                    st.error("Item name is required.")
                else:
                    # Create new item dictionary
                    new_item = {
                        'category': category,
                        'item': item_name,
                        'quantity': quantity,
                        'unit': unit,
                        'price': price,
                        'value': value,
                        'received_date': received_date.strftime('%Y-%m-%d'),
                        'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                        'days_until_expiry': shelf_life
                    }
                    
                    # Add to inventory
                    inventory_data = st.session_state.inventory_data
                    inventory_data.append(new_item)
                    
                    # Update session state and save
                    st.session_state.inventory_data = inventory_data
                    save_inventory_data(inventory_data)
                    
                    st.success(f"Added {quantity} {unit} of {item_name} to inventory.")
    
    with edit_col:
        st.subheader("Edit/Remove Items")
        
        # Display inventory items with edit and delete buttons
        inventory_df = pd.DataFrame(st.session_state.inventory_data)
        
        if not inventory_df.empty:
            # Display a simplified view for editing
            edit_view = inventory_df[['category', 'item', 'quantity', 'unit', 'price']].reset_index()
            edited_df = st.data_editor(
                edit_view,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed"
            )
            
            if st.button("Save Changes"):
                # Update the inventory data with edited values
                for i, row in edited_df.iterrows():
                    if i < len(st.session_state.inventory_data):
                        # Update fields that might have changed
                        st.session_state.inventory_data[i]['category'] = row['category']
                        st.session_state.inventory_data[i]['item'] = row['item']
                        st.session_state.inventory_data[i]['quantity'] = row['quantity']
                        st.session_state.inventory_data[i]['unit'] = row['unit']
                        st.session_state.inventory_data[i]['price'] = row['price']
                        
                        # Recalculate value
                        st.session_state.inventory_data[i]['value'] = row['quantity'] * row['price']
                
                # Save updated inventory
                save_inventory_data(st.session_state.inventory_data)
                st.success("Inventory updated successfully!")
        else:
            st.info("No inventory items to edit.")

# Tab 4: Expiry Tracking
with tab4:
    st.header("Expiry Tracking")
    st.write("Monitor items approaching expiry to reduce waste.")
    
    # Convert inventory data to DataFrame
    inventory_df = pd.DataFrame(st.session_state.inventory_data)
    
    if 'expiry_date' in inventory_df.columns:
        # Convert expiry_date to datetime
        inventory_df['expiry_date'] = pd.to_datetime(inventory_df['expiry_date'])
        
        # Calculate days until expiry relative to March 30, 2025
        today = pd.Timestamp(2025, 3, 30)  # Fixed date (March 30, 2025)
        inventory_df['days_until_expiry'] = (inventory_df['expiry_date'] - today).dt.days
        
        # Create expiry categories including expired items
        def expiry_category(days):
            if days <= 0:
                return "Expired"
            elif days <= 3:
                return "Critical (0-3 days)"
            elif days <= 7:
                return "Warning (4-7 days)"
            elif days <= 14:
                return "Approaching (8-14 days)"
            else:
                return "Safe (>14 days)"
        
        inventory_df['expiry_status'] = inventory_df['days_until_expiry'].apply(expiry_category)
        
        # Display expiry summary
        expiry_summary = inventory_df['expiry_status'].value_counts().reset_index()
        expiry_summary.columns = ['Status', 'Count']
        
        # Define color map for expiry status including expired items
        color_map = {
            "Expired": "red",
            "Critical (0-3 days)": "orange",
            "Warning (4-7 days)": "yellow",
            "Approaching (8-14 days)": "blue",
            "Safe (>14 days)": "green"
        }
        
        # Display expiry summary chart
        expiry_summary = expiry_summary.sort_values(by='Status', key=lambda x: 
                                                 pd.Categorical(x, categories=[
                                                     "Expired",
                                                     "Critical (0-3 days)", 
                                                     "Warning (4-7 days)", 
                                                     "Approaching (8-14 days)", 
                                                     "Safe (>14 days)"
                                                 ]))
        
        expiry_chart = px.bar(expiry_summary, x='Status', y='Count',
                           title='Inventory by Expiry Status',
                           color='Status',
                           color_discrete_map=color_map)
        
        st.plotly_chart(expiry_chart, use_container_width=True)
        
        # Display items by expiry status
        status_tabs = st.tabs([
            "Expired",
            "Critical (0-3 days)", 
            "Warning (4-7 days)", 
            "Approaching (8-14 days)",
            "All Items"
        ])
        
        with status_tabs[0]:
            expired = inventory_df[inventory_df['expiry_status'] == "Expired"].sort_values('expiry_date')
            if not expired.empty:
                st.error("⚠️ Expired items should be discarded or used immediately!")
                st.dataframe(expired[['category', 'item', 'quantity', 'unit', 'expiry_date', 'days_until_expiry', 'value']])
            else:
                st.success("No expired items! 🎉")

        with status_tabs[1]:
            critical = inventory_df[inventory_df['expiry_status'] == "Critical (0-3 days)"].sort_values('expiry_date')
            if not critical.empty:
                st.warning("⚠️ These items need immediate attention!")
                st.dataframe(critical[['category', 'item', 'quantity', 'unit', 'expiry_date', 'days_until_expiry', 'value']])
            else:
                st.success("No items in critical expiry range.")
        
        with status_tabs[2]:
            warning = inventory_df[inventory_df['expiry_status'] == "Warning (4-7 days)"].sort_values('expiry_date')
            if not warning.empty:
                st.info("These items should be used in the coming week.")
                st.dataframe(warning[['category', 'item', 'quantity', 'unit', 'expiry_date', 'days_until_expiry', 'value']])
            else:
                st.success("No items in warning expiry range.")
        
        with status_tabs[3]:
            approaching = inventory_df[inventory_df['expiry_status'] == "Approaching (8-14 days)"].sort_values('expiry_date')
            if not approaching.empty:
                st.write("These items are approaching expiry but are still safe for now.")
                st.dataframe(approaching[['category', 'item', 'quantity', 'unit', 'expiry_date', 'days_until_expiry', 'value']])
            else:
                st.success("No items approaching expiry.")
        
        with status_tabs[4]:
            st.write("All inventory items sorted by expiry date:")
            st.dataframe(inventory_df.sort_values('expiry_date')[['category', 'item', 'quantity', 'unit', 'expiry_date', 'days_until_expiry', 'expiry_status', 'value']])
        
        # Calculate potential waste value - include both expired and critical items
        expired_value = expired['value'].sum() if not expired.empty else 0
        critical_value = critical['value'].sum() if not critical.empty else 0
        
        st.metric("Potential Waste Value", f"₹{expired_value + critical_value:.2f}",
                 delta=f"-{expired_value:.2f}" if expired_value > 0 else None,
                 delta_color="inverse")
        
        # Recommendations based on expiry status - using both expired and critical items
        st.subheader("Recommendations")
        
        if not expired.empty or not critical.empty:
            combined = pd.concat([expired, critical]) if not expired.empty and not critical.empty else (expired if not expired.empty else critical)
            items_to_use = combined.sort_values('days_until_expiry')['item'].tolist()
            
            st.write("Consider using these items immediately:")
            for i, item in enumerate(items_to_use[:5]):  # Show top 5
                st.write(f"{i+1}. {item}")
            
            if len(items_to_use) > 5:
                st.write(f"... and {len(items_to_use) - 5} more items")
        else:
            st.success("Your inventory is in good shape! No items need immediate attention.")
    else:
        st.info("No expiry date information available in the inventory.")
