import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from datetime import datetime, timedelta, date
import jdatetime  # For Jalali to Gregorian conversion

# Function to convert Jalali dates to Gregorian
@st.cache_data
def jalali_to_gregorian_vectorized(date_series):
    def convert(date_str):
        if pd.isna(date_str):
            return pd.NaT
        try:
            year, month, day = map(int, date_str.split('/'))
            return jdatetime.date(year, month, day).togregorian()
        except:
            return pd.NaT
    return date_series.apply(convert)

# Function to extract VIP status based on emojis
@st.cache_data
def extract_vip_status(name_series):
    def get_vip_status(name):
        if pd.isna(name):
            return 'Non-VIP'
        if '💎' in name:
            return 'Gold VIP'
        elif '⭐' in name:
            return 'Silver VIP'
        elif '💠' in name:
            return 'Bronze VIP'
        else:
            return 'Non-VIP'
    return name_series.apply(get_vip_status)

# Function to load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    # Load the Excel file
    data = pd.read_excel(uploaded_file)

    # List of columns containing Jalali dates
    date_columns = [
        'تاریخ انجام معامله', 'تاریخ ایجاد معامله', 'تاریخ احتمالی انجام معامله',
        'تاریخ ورود', 'تاریخ خروج', 'شروع قرارداد', 'پایان قرارداد'
        # Add any additional date columns here
    ]

    # Convert Jalali dates to Gregorian
    for col in date_columns:
        if col in data.columns:
            data[col] = jalali_to_gregorian_vectorized(data[col])
            # Ensure the date columns are datetime objects
            data[col] = pd.to_datetime(data[col], errors='coerce')
        else:
            st.warning(f"Column '{col}' not found in the data.")

    # Clean 'تعداد شب ' column by removing non-digit characters
    if 'تعداد شب ' in data.columns:
        data['تعداد شب '] = data['تعداد شب '].astype(str).str.replace(r'\D', '', regex=True)
        data['تعداد شب '] = pd.to_numeric(data['تعداد شب '], errors='coerce')
        # Remove entries where 'تعداد شب ' is unreasonably large (e.g., greater than 365)
        data.loc[data['تعداد شب '] > 365, 'تعداد شب '] = np.nan
    else:
        st.warning("Column 'تعداد شب ' not found in the data.")

    # Similarly, convert 'ارزش معامله' to numeric
    if 'ارزش معامله' in data.columns:
        data['ارزش معامله'] = pd.to_numeric(data['ارزش معامله'], errors='coerce')
    else:
        st.warning("Column 'ارزش معامله' not found in the data.")

    # Extract VIP Status
    data['VIP Status'] = extract_vip_status(data['نام خانوادگی شخص معامله'])

    return data

# Function to calculate RFM
@st.cache_data
def calculate_rfm(data, today=None):
    # Divide 'ارزش معامله' by 10 as per the new requirement
    data['ارزش معامله'] = data['ارزش معامله'] / 10

    # Filter for successful deals
    successful_deals = data[data['وضعیت معامله'] == 'موفق']

    # Define today's date for recency calculation
    if today is None:
        today = datetime.today()
    else:
        today = pd.to_datetime(today)

    # Group by unique customer ID while including personal details
    rfm_data = successful_deals.groupby('کد دیدار شخص معامله').agg({
        'نام شخص معامله': 'first',
        'نام خانوادگی شخص معامله': 'first',
        'موبایل شخص معامله': 'first',
        'تاریخ انجام معامله': lambda x: (today - pd.to_datetime(x).max()).days,  # Recency
        'کد دیدار معامله': 'count',  # Frequency
        'ارزش معامله': 'sum',  # Monetary
        'VIP Status': 'first'  # VIP Status
    }).reset_index()

    # Rename columns for clarity
    rfm_data.rename(columns={
        'کد دیدار شخص معامله': 'Customer ID',
        'نام شخص معامله': 'First Name',
        'نام خانوادگی شخص معامله': 'Last Name',
        'موبایل شخص معامله': 'Phone Number',
        'تاریخ انجام معامله': 'Recency',
        'کد دیدار معامله': 'Frequency',
        'ارزش معامله': 'Monetary',
    }, inplace=True)

    return rfm_data

# Function for RFM segmentation
@st.cache_data
def rfm_segmentation(data):
    data = data[(data['Monetary'] > 0) & (data['Customer ID'] != 0)]
    # Define R, F, M thresholds based on quantiles to categorize scores
    buckets = data[['Recency', 'Frequency', 'Monetary']].quantile([1/3, 2/3]).to_dict()

    # Define the RFM segmentation function
    def rfm_segment(row):
        # Recency scoring
        if row['Recency'] >= 296:
            r_score = 1
        elif row['Recency'] >= 185:
            r_score = 2
        elif row['Recency'] >= 76:
            r_score = 3
        else:
            r_score = 4

        # Frequency scoring based on quantiles
        if row['Frequency'] <= buckets['Frequency'][1/3]:
            f_score = 1
        elif row['Frequency'] <= buckets['Frequency'][2/3]:
            f_score = 2
        else:
            f_score = 3

        # Monetary scoring based on quantiles
        if row['Monetary'] <= buckets['Monetary'][1/3]:
            m_score = 1
        elif row['Monetary'] <= buckets['Monetary'][2/3]:
            m_score = 2
        else:
            m_score = 3

        return f"{r_score}{f_score}{m_score}"

    # Apply the segmentation function to categorize customers into RFM segments
    data['RFM_segment'] = data.apply(rfm_segment, axis=1)

    # Define segment labels based on RFM combinations
    segment_labels = {
        '111': 'Churned',
        '112': 'Churned',
        '113': 'Lost Big Spenders',
        '121': 'Churned',
        '122': 'Churned',
        '123': 'Lost Big Spenders',
        '131': 'Hibernating',
        '132': 'Big Loss',
        '133': 'Big Loss',
        '211': 'Low Value',
        '212': 'At Risk',
        '213': 'At Risk',
        '221': 'Low Value',
        '222': 'At Risk',
        '223': 'At Risk',
        '231': 'At Risk',
        '232': 'At Risk',
        '233': 'At Risk',
        '311': 'Low Value',
        '312': 'Promising',
        '313': 'Big Spenders',
        '321': 'Promising',
        '322': 'Promising',
        '323': 'Promising',
        '331': 'Loyal Customers',
        '332': 'Loyal Customers',
        '333': 'Loyal Customers',
        '411': 'Promising',
        '412': 'Promising',
        '413': 'Big Spenders',
        '421': 'Price Sensitive',
        '422': 'Loyal Customers',
        '423': 'Loyal Customers',
        '431': 'Price Sensitive',
        '432': 'Loyal Customers',
        '433': 'Champions'
    }

    # Map the segment label to each RFM segment
    data['RFM_segment_label'] = data['RFM_segment'].map(segment_labels)
    return data

# Function to normalize RFM values for plotting
@st.cache_data
def normalize_rfm(data):
    scaler = MinMaxScaler()

    # For Recency, invert the scale so that higher is better (more recent purchase)
    data['Recency_norm'] = scaler.fit_transform(data[['Recency']])
    data['Recency_norm'] = 1 - data['Recency_norm']  # Invert Recency scores

    # Normalize Frequency and Monetary normally
    data[['Frequency_norm', 'Monetary_norm']] = scaler.fit_transform(
        data[['Frequency', 'Monetary']]
    )
    return data

# Global functions for data conversion (moved outside conditional blocks)
@st.cache_data
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def main():
    # Set page config
    st.set_page_config(
        page_title="داشبورد سگمنت‌بندی مشتریان تهران‌مبله",
        page_icon="📊",
        layout="wide",
    )

    # Title
    st.title("Customer Segmentation Dashboard - Tehran Moble")

    # File uploader
    st.sidebar.header("Upload your deals Excel file")
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Load and preprocess data
            data_load_state = st.text('Loading and processing data...')
            data = load_data(uploaded_file)

            # Get unique options for filters
            product_options = data['عنوان محصول'].dropna().unique().tolist()
            product_options.sort()

            sellers_options = data['مسئول معامله'].dropna().unique().tolist()
            sellers_options.sort()

            sale_channels_options = data['شیوه آشنایی معامله'].dropna().unique().tolist()
            sale_channels_options.sort()

            vip_options = data['VIP Status'].dropna().unique().tolist()
            vip_options.sort()

            # ------------------ Navigation ------------------
            st.sidebar.header("Navigation")
            page = st.sidebar.radio("Go to", ['General', 'Compare RFM Segments Over Time', 'Portfolio Analysis', 'Seller Analysis', 'Sale Channel Analysis', 'VIP Analysis', 'Customer Inquiry Module'])

            # ------------------ Global Filters ------------------
            st.sidebar.header("Global Filters")

            # Multiselect for VIP Status
            select_all_vips = st.sidebar.checkbox("Select all VIP statuses", value=True, key='select_all_vips_global')

            if select_all_vips:
                selected_vips = vip_options
            else:
                selected_vips = st.sidebar.multiselect(
                    "Select VIP Status:",
                    options=vip_options,
                    default=[],
                    key='vips_multiselect_global'
                )

            # Multiselect for Products
            select_all_products_global = st.sidebar.checkbox("Select all products", value=True, key='select_all_products_global')

            if select_all_products_global:
                selected_products_global = product_options
            else:
                selected_products_global = st.sidebar.multiselect(
                    "Select Products (عنوان محصول):",
                    options=product_options,
                    default=[],
                    key='products_multiselect_global'
                )

            # Multiselect for Sellers
            select_all_sellers = st.sidebar.checkbox("Select all sellers", value=True, key='select_all_sellers_global')

            if select_all_sellers:
                selected_sellers = sellers_options
            else:
                selected_sellers = st.sidebar.multiselect(
                    "Select Sellers :",
                    options=sellers_options,
                    default=[],
                    key='sellers_multiselect_global'
                )

            # Multiselect for Sale Channels
            select_all_sale_channels = st.sidebar.checkbox("Select all sale channels", value=True, key='select_all_sale_channels_global')

            if select_all_sale_channels:
                selected_sale_channels = sale_channels_options
            else:
                selected_sale_channels = st.sidebar.multiselect(
                    "Select Sale Channels (شیوه آشنایی معامله):",
                    options=sale_channels_options,
                    default=[],
                    key='sale_channels_multiselect_global'
                )

            # Apply Global Filters
            filtered_data = data.copy()

            # Apply VIP Filter
            if selected_vips:
                filtered_data = filtered_data[filtered_data['VIP Status'].isin(selected_vips)]
            else:
                st.warning("No VIP statuses selected. Please select at least one VIP status.")
                filtered_data = pd.DataFrame()

            # Apply Product Filter
            if selected_products_global:
                filtered_data = filtered_data[filtered_data['عنوان محصول'].isin(selected_products_global)]
            else:
                st.warning("No products selected. Please select at least one product.")
                filtered_data = pd.DataFrame()

            # Apply Seller Filter
            if selected_sellers:
                filtered_data = filtered_data[filtered_data['مسئول معامله'].isin(selected_sellers)]
            else:
                st.warning("No sellers selected. Please select at least one seller.")
                filtered_data = pd.DataFrame()

            # Apply Sale Channel Filter
            if selected_sale_channels:
                filtered_data = filtered_data[filtered_data['شیوه آشنایی معامله'].isin(selected_sale_channels)]
            else:
                st.warning("No sale channels selected. Please select at least one sale channel.")
                filtered_data = pd.DataFrame()

            if filtered_data.empty:
                st.error("No data available after applying the global filters.")
                st.stop()

            # Cache the filtered data
            @st.cache_data
            def get_filtered_data():
                return filtered_data.copy()

            filtered_data = get_filtered_data()

            # Ensure 'تاریخ انجام معامله' is datetime and handle NaT
            filtered_data['تاریخ انجام معامله'] = pd.to_datetime(filtered_data['تاریخ انجام معامله'], errors='coerce')
            filtered_data = filtered_data.dropna(subset=['تاریخ انجام معامله'])

            # Calculate RFM (Current RFM) on entire data
            rfm_data = calculate_rfm(data)
            rfm_data = rfm_segmentation(rfm_data)
            rfm_data = normalize_rfm(rfm_data)
            data_load_state.text('Loading and processing data...done!')

            # Define colors for segments (used globally)
            COLOR_MAP = {
                "Champions": "#00CC96",            # Green
                "Loyal Customers": "#19D3F3",      # Light Blue
                "Promising": "#B6E880",            # Light Green
                "Big Spenders": "#FF6692",         # Pink
                "Price Sensitive": "#FFA15A",      # Orange
                "At Risk": "#AB63FA",              # Purple
                "Churned": "#c21e56",              # Red
                "Hibernating": "#636EFA",          # Blue
                "Lost Big Spenders": "#FF7415",    # Orange-Red
                "Big Loss": "#cdca49",             # Olive/Khaki
                "Low Value": "#D3D3D3",            # Gray
            }

            # Filter RFM data based on customers in filtered_data
            customers_in_filtered_data = filtered_data['کد دیدار شخص معامله'].unique()
            rfm_data_filtered_global = rfm_data[rfm_data['Customer ID'].isin(customers_in_filtered_data)]

            # ------------------ Pages ------------------

            if page == 'General':
                # ------------------ Segment Filters for RFM Plots ------------------

                st.subheader("Filter RFM Plots by Segments")

                # VIP Filter for this page
                vip_options_page = sorted(rfm_data_filtered_global['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses for plots", value=True, key='select_all_vips_plots')

                if select_all_vips_page:
                    selected_vips_plots = vip_options_page
                else:
                    selected_vips_plots = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=[],
                        key='vips_multiselect_plots'
                    )

                rfm_data_filtered_global = rfm_data_filtered_global[rfm_data_filtered_global['VIP Status'].isin(selected_vips_plots)]

                segment_options = sorted(rfm_data_filtered_global['RFM_segment_label'].unique())
                select_all_segments = st.checkbox("Select all segments", value=True, key='select_all_segments_plots')

                if select_all_segments:
                    selected_segments_plots = segment_options
                else:
                    selected_segments_plots = st.multiselect(
                        "Select RFM Segments:",
                        options=segment_options,
                        default=[],
                        key='segments_multiselect_plots'
                    )

                if selected_segments_plots:
                    rfm_data_filtered_plots = rfm_data_filtered_global[rfm_data_filtered_global['RFM_segment_label'].isin(selected_segments_plots)]
                else:
                    st.warning("No segments selected. Please select at least one segment.")
                    rfm_data_filtered_plots = pd.DataFrame()

                if rfm_data_filtered_plots.empty:
                    st.warning("No data available for the selected segments and VIP statuses.")
                else:
                    # Pie chart of RFM segments
                    st.subheader("Distribution of RFM Segments")
                    rfm_segment_counts = rfm_data_filtered_plots['RFM_segment_label'].value_counts().reset_index()
                    rfm_segment_counts.columns = ['RFM_segment_label', 'Count']

                    fig_pie = px.pie(
                        rfm_segment_counts,
                        names='RFM_segment_label',
                        values='Count',
                        color='RFM_segment_label',
                        color_discrete_map=COLOR_MAP,
                        hole=0.4
                    )

                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

                    st.plotly_chart(fig_pie)

                    # 3D Scatter Plot
                    st.subheader("3D Scatter Plot of RFM Segments")

                    fig_3d = px.scatter_3d(
                        rfm_data_filtered_plots,
                        x='Recency_norm',
                        y='Frequency_norm',
                        z='Monetary_norm',
                        color='RFM_segment_label',
                        color_discrete_map=COLOR_MAP,
                        hover_data=['Customer ID', 'First Name', 'Last Name', 'VIP Status'],
                        title='RFM Segments in Normalized Space'
                    )

                    fig_3d.update_layout(
                        scene=dict(
                            xaxis_title='Recency (Higher is Better)',
                            yaxis_title='Frequency',
                            zaxis_title='Monetary Value'
                        ),
                        legend_title='RFM Segments'
                    )

                    st.plotly_chart(fig_3d)

                    # Additional plots
                    st.subheader("RFM Metrics Distribution")

                    # Histogram for Recency
                    fig_recency = px.histogram(
                        rfm_data_filtered_plots,
                        x='Recency',
                        nbins=50,
                        title='Recency Distribution',
                        color='RFM_segment_label',
                        color_discrete_map=COLOR_MAP
                    )
                    st.plotly_chart(fig_recency)

                    # Histogram for Frequency
                    fig_frequency = px.histogram(
                        rfm_data_filtered_plots,
                        x='Frequency',
                        nbins=50,
                        title='Frequency Distribution',
                        color='RFM_segment_label',
                        color_discrete_map=COLOR_MAP
                    )
                    st.plotly_chart(fig_frequency)

                    # Histogram for Monetary
                    fig_monetary = px.histogram(
                        rfm_data_filtered_plots,
                        x='Monetary',
                        nbins=50,
                        title='Monetary Value Distribution',
                        labels={'Monetary': 'Monetary Value'},
                        color='RFM_segment_label',
                        color_discrete_map=COLOR_MAP
                    )
                    st.plotly_chart(fig_monetary)

                # ------------------ Customer Segmentation Data Table ------------------

                st.subheader("Customer Segmentation Data")

                # Product Filter for Data Table
                st.subheader("Filter Table by Products")
                select_all_products_table = st.checkbox("Select all products", value=True, key='select_all_products_table')

                if select_all_products_table:
                    selected_products_table = product_options
                else:
                    selected_products_table = st.multiselect(
                        "Select products (عنوان محصول):",
                        options=product_options,
                        default=[],
                        key='products_multiselect_table'
                    )

                if selected_products_table:
                    # Get customer IDs who have purchased the selected products
                    customers_with_selected_products = data[data['عنوان محصول'].isin(selected_products_table)]['کد دیدار شخص معامله'].unique()
                    # Filter RFM data
                    rfm_data_filtered_table = rfm_data_filtered_global[rfm_data_filtered_global['Customer ID'].isin(customers_with_selected_products)]
                else:
                    rfm_data_filtered_table = rfm_data_filtered_global.copy()

                st.write(rfm_data_filtered_table[['Customer ID', 'First Name', 'Last Name', 'VIP Status', 'Phone Number', 'Recency', 'Frequency', 'Monetary', 'RFM_segment_label']])

                # Optionally, allow users to download the data
                csv_data = convert_df(rfm_data_filtered_table)
                excel_data = convert_df_to_excel(rfm_data_filtered_table)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download data as CSV",
                        data=csv_data,
                        file_name='rfm_segmentation.csv',
                        mime='text/csv',
                    )
                with col2:
                    st.download_button(
                        label="Download data as Excel",
                        data=excel_data,
                        file_name='rfm_segmentation.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )

            elif page == 'Compare RFM Segments Over Time':
                # ------------------ Compare RFM Segments Over Time ------------------

                st.subheader("Compare RFM Segments Over Time")

                # VIP Filter for this page
                vip_options_page = sorted(rfm_data['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses for comparison", value=True, key='select_all_vips_comparison')

                if select_all_vips_page:
                    selected_vips_comparison = vip_options_page
                else:
                    selected_vips_comparison = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=[],
                        key='vips_multiselect_comparison'
                    )

                # Use a form to prevent automatic reruns
                with st.form(key='comparison_form'):
                    # Date Input
                    comparison_date = st.date_input("Select a date for comparison", value=datetime.today())

                    # Ensure that the date is not in the future
                    if comparison_date > datetime.today().date():
                        st.error("The comparison date cannot be in the future.")
                        submit_button = st.form_submit_button(label='Submit')
                    else:
                        # Get list of unique segments
                        segment_options = ['All'] + sorted(rfm_data['RFM_segment_label'].dropna().unique())

                        col1, col2 = st.columns(2)
                        with col1:
                            from_segment = st.selectbox("Select 'FROM' Segment (RFM1)", options=segment_options)
                        with col2:
                            to_segment = st.selectbox("Select 'TO' Segment (RFM2)", options=segment_options)

                        # Show Results button
                        submit_button = st.form_submit_button(label='Show Results')

                if 'submit_button' in locals() and submit_button:
                    # Filter data before the selected date
                    data_before_date = data[data['تاریخ انجام معامله'] <= pd.to_datetime(comparison_date)]

                    if data_before_date.empty:
                        st.warning("No data available before the selected date.")
                    else:
                        # Calculate RFM1 (RFM before the selected date)
                        rfm_data1 = calculate_rfm(data_before_date, today=comparison_date)
                        rfm_data1 = rfm_segmentation(rfm_data1)

                        # Filter RFM data based on selected VIP statuses
                        rfm_data1 = rfm_data1[rfm_data1['VIP Status'].isin(selected_vips_comparison)]
                        rfm_data_filtered = rfm_data[rfm_data['VIP Status'].isin(selected_vips_comparison)]

                        # Prepare data for comparison
                        # Merge RFM1 and RFM2 on 'Customer ID'
                        comparison_df = rfm_data1[['Customer ID', 'First Name', 'Last Name', 'Phone Number', 'VIP Status', 'RFM_segment_label']].merge(
                            rfm_data_filtered[['Customer ID', 'RFM_segment_label']],
                            on='Customer ID',
                            how='inner',
                            suffixes=('_RFM1', '_RFM2')
                        )

                        # Handle the cases
                        if from_segment == 'All' and to_segment == 'All':
                            st.error("Please select at least one segment in 'FROM' or 'TO'.")
                        else:
                            if from_segment != 'All':
                                comparison_df = comparison_df[comparison_df['RFM_segment_label_RFM1'] == from_segment]
                            if to_segment != 'All':
                                comparison_df = comparison_df[comparison_df['RFM_segment_label_RFM2'] == to_segment]

                            if comparison_df.empty:
                                st.warning("No customers found for the selected segment transitions and VIP statuses.")
                            else:
                                # Display count and bar chart
                                counts = comparison_df['RFM_segment_label_RFM2'].value_counts().reset_index()
                                counts.columns = ['RFM_segment_label_RFM2', 'Count']

                                st.write(f"Number of customers matching the criteria: **{len(comparison_df)}**")

                                fig = px.bar(
                                    counts,
                                    x='RFM_segment_label_RFM2',
                                    y='Count',
                                    color='RFM_segment_label_RFM2',
                                    color_discrete_map=COLOR_MAP,
                                    text='Count',
                                    labels={'RFM_segment_label_RFM2': 'RFM2 Segment', 'Count': 'Number of Customers'}
                                )
                                fig.update_traces(textposition='outside')
                                st.plotly_chart(fig)

                                # Show customer table
                                st.subheader("Customer Details")
                                customer_table = comparison_df[['Customer ID', 'First Name', 'Last Name', 'Phone Number', 'VIP Status', 'RFM_segment_label_RFM1', 'RFM_segment_label_RFM2']]
                                customer_table.rename(columns={
                                    'RFM_segment_label_RFM1': 'RFM1 Segment',
                                    'RFM_segment_label_RFM2': 'RFM2 Segment'
                                }, inplace=True)
                                st.write(customer_table)

                                # Download buttons
                                csv_data = convert_df(customer_table)
                                excel_data = convert_df_to_excel(customer_table)

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="Download data as CSV",
                                        data=csv_data,
                                        file_name='rfm_segment_comparison.csv',
                                        mime='text/csv',
                                    )
                                with col2:
                                    st.download_button(
                                        label="Download data as Excel",
                                        data=excel_data,
                                        file_name='rfm_segment_comparison.xlsx',
                                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                    )

            elif page == 'Portfolio Analysis':
                # ------------------ Portfolio Analysis ------------------

                st.subheader("Portfolio Analysis by Cluster and Product")

                # Get unique clusters from RFM data
                cluster_options = rfm_data['RFM_segment_label'].unique().tolist()
                cluster_options.sort()

                # Move checkboxes outside the form to ensure they trigger app reruns
                select_all_clusters = st.checkbox("Select all clusters", value=True, key='select_all_clusters_portfolio')

                select_all_products_portfolio = st.checkbox("Select all products", value=True, key='select_all_products_portfolio')

                # VIP Filter for this page
                vip_options_page = sorted(rfm_data['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_portfolio')

                # Create a form to prevent automatic reruns
                with st.form(key='portfolio_form'):
                    # Multiselect for clusters
                    if select_all_clusters:
                        selected_clusters = cluster_options
                    else:
                        selected_clusters = st.multiselect(
                            "Select Clusters:",
                            options=cluster_options,
                            default=[],
                            key='clusters_multiselect_portfolio'
                        )

                    # Multiselect for products
                    if select_all_products_portfolio:
                        selected_products_portfolio = product_options
                    else:
                        selected_products_portfolio = st.multiselect(
                            "Select Products (عنوان محصول):",
                            options=product_options,
                            default=[],
                            key='products_multiselect_portfolio'
                        )

                    # VIP statuses
                    if select_all_vips_page:
                        selected_vips_portfolio = vip_options_page
                    else:
                        selected_vips_portfolio = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_page,
                            default=[],
                            key='vips_multiselect_portfolio'
                        )

                    # Apply button
                    apply_portfolio = st.form_submit_button(label='Apply')

                if apply_portfolio:
                    if not selected_clusters:
                        st.warning("Please select at least one cluster.")
                    else:
                        if not selected_products_portfolio:
                            st.warning("Please select at least one product.")
                        else:
                            if not selected_vips_portfolio:
                                st.warning("Please select at least one VIP status.")
                            else:
                                # Get customers in selected clusters and VIP statuses
                                customers_in_clusters = rfm_data[(rfm_data['RFM_segment_label'].isin(selected_clusters)) &
                                                                 (rfm_data['VIP Status'].isin(selected_vips_portfolio))]['Customer ID'].unique()

                                # Filter deals data for these customers and selected products
                                deals_filtered = data[data['کد دیدار شخص معامله'].isin(customers_in_clusters)]
                                deals_filtered = deals_filtered[deals_filtered['عنوان محصول'].isin(selected_products_portfolio)]

                                # Apply global filters
                                deals_filtered = deals_filtered[
                                    (deals_filtered['عنوان محصول'].isin(selected_products_global)) &
                                    (deals_filtered['مسئول معامله'].isin(selected_sellers)) &
                                    (deals_filtered['شیوه آشنایی معامله'].isin(selected_sale_channels))
                                ]

                                if deals_filtered.empty:
                                    st.warning("No deals found for the selected clusters, VIP statuses, and products.")
                                else:
                                    # Calculate distributions
                                    # Frequency: Number of times each product was bought
                                    frequency_distribution = deals_filtered.groupby('عنوان محصول').size().reset_index(name='Frequency')

                                    # Monetary: Total money spent on each product
                                    monetary_distribution = deals_filtered.groupby('عنوان محصول')['ارزش معامله'].sum().reset_index()

                                    # Plot Frequency Distribution
                                    st.subheader("Frequency Distribution of Products")
                                    fig_freq = px.bar(
                                        frequency_distribution,
                                        x='عنوان محصول',
                                        y='Frequency',
                                        title='Frequency Distribution',
                                        labels={'عنوان محصول': 'Product', 'Frequency': 'Number of Purchases'},
                                        text='Frequency'
                                    )
                                    fig_freq.update_traces(textposition='outside')
                                    st.plotly_chart(fig_freq)

                                    # Plot Monetary Distribution
                                    st.subheader("Monetary Distribution of Products")
                                    fig_monetary = px.bar(
                                        monetary_distribution,
                                        x='عنوان محصول',
                                        y='ارزش معامله',
                                        title='Monetary Distribution',
                                        labels={'عنوان محصول': 'Product', 'ارزش معامله': 'Total Monetary Value'},
                                        text='ارزش معامله'
                                    )
                                    fig_monetary.update_traces(textposition='outside')
                                    st.plotly_chart(fig_monetary)

                                    # ------------------ Customer Table ------------------

                                    st.subheader("Customer Details")

                                    # Get successful deals
                                    successful_deals = deals_filtered[deals_filtered['وضعیت معامله'] == 'موفق']

                                    # Sum of 'تعداد شب ' per customer per product
                                    customer_nights = successful_deals.groupby(['کد دیدار شخص معامله', 'عنوان محصول'])['تعداد شب '].sum().unstack(fill_value=0)

                                    # Merge with RFM data
                                    customer_details = rfm_data[rfm_data['Customer ID'].isin(customers_in_clusters)][['Customer ID', 'First Name', 'Last Name', 'VIP Status', 'RFM_segment_label', 'Recency', 'Frequency', 'Monetary']]
                                    customer_details = customer_details.merge(customer_nights, left_on='Customer ID', right_index=True, how='left').fillna(0)

                                    st.write(customer_details)

                                    # Download buttons
                                    csv_data = convert_df(customer_details)
                                    excel_data = convert_df_to_excel(customer_details)

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.download_button(
                                            label="Download data as CSV",
                                            data=csv_data,
                                            file_name='portfolio_analysis.csv',
                                            mime='text/csv',
                                        )
                                    with col2:
                                        st.download_button(
                                            label="Download data as Excel",
                                            data=excel_data,
                                            file_name='portfolio_analysis.xlsx',
                                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                        )

            elif page == 'Seller Analysis':
                # ------------------ Seller Analysis Module ------------------

                st.subheader("Seller Analysis")

                analysis_option = st.radio("Select Analysis Type:", options=['By Seller', 'By Cluster'], key='seller_analysis_option')

                # VIP Filter for this page
                vip_options_page = sorted(rfm_data['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_seller')

                if select_all_vips_page:
                    selected_vips_seller = vip_options_page
                else:
                    selected_vips_seller = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=[],
                        key='vips_multiselect_seller'
                    )

                if analysis_option == 'By Seller':
                    with st.form(key='seller_analysis_form'):
                        selected_seller = st.selectbox("Select a Seller:", options=sellers_options)
                        # Date Range Input
                        min_date = data['تاریخ انجام معامله'].min()
                        max_date = data['تاریخ انجام معامله'].max()

                        # Ensure min_date and max_date are dates
                        if pd.isna(min_date) or pd.isna(max_date):
                            st.warning("Date range is invalid. Please check your data.")
                            return

                        min_date = min_date.date()
                        max_date = max_date.date()

                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key='seller_start_date')
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key='seller_end_date')

                        # Apply Filters button
                        apply_seller_filters = st.form_submit_button(label='Apply Filters')

                    if apply_seller_filters:
                        if selected_seller:
                            if selected_vips_seller:
                                # Filter data based on date range and successful deals
                                date_filtered_data = data[
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date)) &
                                    (data['وضعیت معامله'] == 'موفق')
                                ]

                                # Apply global filters
                                date_filtered_data = date_filtered_data[
                                    (date_filtered_data['عنوان محصول'].isin(selected_products_global)) &
                                    (date_filtered_data['مسئول معامله'].isin(selected_sellers)) &
                                    (date_filtered_data['شیوه آشنایی معامله'].isin(selected_sale_channels)) &
                                    (date_filtered_data['VIP Status'].isin(selected_vips_seller))
                                ]

                                seller_data = date_filtered_data[date_filtered_data['مسئول معامله'] == selected_seller]
                                if seller_data.empty:
                                    st.warning("No successful deals found for the selected seller and VIP statuses in the specified date range.")
                                else:
                                    # Get customer IDs
                                    seller_customer_ids = seller_data['کد دیدار شخص معامله'].unique()
                                    seller_rfm_data = rfm_data[rfm_data['Customer ID'].isin(seller_customer_ids)]

                                    if seller_rfm_data.empty:
                                        st.warning("No RFM data available for the selected seller and VIP statuses.")
                                    else:
                                        # Count of customers in each cluster
                                        cluster_counts = seller_rfm_data['RFM_segment_label'].value_counts().reset_index()
                                        cluster_counts.columns = ['RFM_segment_label', 'Count']

                                        # Plot frequency chart
                                        fig_seller_freq = px.bar(
                                            cluster_counts,
                                            x='RFM_segment_label',
                                            y='Count',
                                            title=f"Cluster Distribution (Frequency) for Seller: {selected_seller}",
                                            labels={'RFM_segment_label': 'RFM Segment', 'Count': 'Number of Customers'},
                                            text='Count',
                                            color='RFM_segment_label',
                                            color_discrete_map=COLOR_MAP
                                        )
                                        fig_seller_freq.update_traces(textposition='outside')
                                        st.plotly_chart(fig_seller_freq)

                                        # Monetary value per cluster
                                        seller_monetary = seller_rfm_data.groupby('RFM_segment_label')['Monetary'].sum().reset_index()
                                        fig_seller_monetary = px.bar(
                                            seller_monetary,
                                            x='RFM_segment_label',
                                            y='Monetary',
                                            title=f"Cluster Distribution (Monetary) for Seller: {selected_seller}",
                                            labels={'RFM_segment_label': 'RFM Segment', 'Monetary': 'Total Monetary Value'},
                                            text='Monetary',
                                            color='RFM_segment_label',
                                            color_discrete_map=COLOR_MAP
                                        )
                                        fig_seller_monetary.update_traces(textposition='outside')
                                        st.plotly_chart(fig_seller_monetary)

                                        # ------------------ Customer Table ------------------

                                        st.subheader("Customer Details")

                                        # Sum of 'تعداد شب ' per customer
                                        customer_nights = seller_data.groupby('کد دیدار شخص معامله')['تعداد شب '].sum().reset_index()
                                        customer_nights.rename(columns={'کد دیدار شخص معامله': 'Customer ID', 'تعداد شب ': 'Total Nights'}, inplace=True)

                                        # Merge with RFM data
                                        customer_details = seller_rfm_data[['Customer ID', 'First Name', 'Last Name', 'VIP Status', 'RFM_segment_label', 'Recency', 'Frequency', 'Monetary']]
                                        customer_details = customer_details.merge(customer_nights, on='Customer ID', how='left').fillna(0)

                                        st.write(customer_details)

                                        # Download buttons
                                        csv_data = convert_df(customer_details)
                                        excel_data = convert_df_to_excel(customer_details)

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.download_button(
                                                label="Download data as CSV",
                                                data=csv_data,
                                                file_name='seller_analysis.csv',
                                                mime='text/csv',
                                            )
                                        with col2:
                                            st.download_button(
                                                label="Download data as Excel",
                                                data=excel_data,
                                                file_name='seller_analysis.xlsx',
                                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                            )
                            else:
                                st.warning("Please select at least one VIP status.")
                        else:
                            st.warning("Please select a seller.")

                elif analysis_option == 'By Cluster':
                    # Move checkbox outside the form
                    select_all_clusters_seller = st.checkbox("Select all clusters", value=True, key='select_all_clusters_seller')

                    with st.form(key='seller_cluster_form'):
                        cluster_options = rfm_data['RFM_segment_label'].unique().tolist()
                        cluster_options.sort()

                        if select_all_clusters_seller:
                            selected_clusters_seller = cluster_options
                        else:
                            selected_clusters_seller = st.multiselect(
                                "Select Clusters:",
                                options=cluster_options,
                                default=[],
                                key='clusters_multiselect_seller'
                            )
                        # Date Range Input
                        min_date = data['تاریخ انجام معامله'].min()
                        max_date = data['تاریخ انجام معامله'].max()

                        # Ensure min_date and max_date are dates
                        if pd.isna(min_date) or pd.isna(max_date):
                            st.warning("Date range is invalid. Please check your data.")
                            return

                        min_date = min_date.date()
                        max_date = max_date.date()

                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key='seller_cluster_start_date')
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key='seller_cluster_end_date')

                        # Apply Filters button
                        apply_cluster_filters = st.form_submit_button(label='Apply Filters')

                    if apply_cluster_filters:
                        if selected_clusters_seller:
                            if selected_vips_seller:
                                # Filter data based on date range and successful deals
                                date_filtered_data = data[
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date)) &
                                    (data['وضعیت معامله'] == 'موفق')
                                ]

                                # Apply global filters
                                date_filtered_data = date_filtered_data[
                                    (date_filtered_data['عنوان محصول'].isin(selected_products_global)) &
                                    (date_filtered_data['مسئول معامله'].isin(selected_sellers)) &
                                    (date_filtered_data['شیوه آشنایی معامله'].isin(selected_sale_channels)) &
                                    (date_filtered_data['VIP Status'].isin(selected_vips_seller))
                                ]

                                cluster_customers = rfm_data[rfm_data['RFM_segment_label'].isin(selected_clusters_seller)]['Customer ID'].unique()
                                cluster_deals = date_filtered_data[date_filtered_data['کد دیدار شخص معامله'].isin(cluster_customers)]

                                if cluster_deals.empty:
                                    st.warning("No successful deals found for the selected clusters and VIP statuses in the specified date range.")
                                else:
                                    # Frequency of deals per seller
                                    seller_counts = cluster_deals['مسئول معامله'].value_counts().reset_index()
                                    seller_counts.columns = ['Seller', 'Count']

                                    fig_seller_cluster_freq = px.bar(
                                        seller_counts,
                                        x='Seller',
                                        y='Count',
                                        title=f"Seller Distribution (Frequency) for Clusters: {', '.join(selected_clusters_seller)}",
                                        labels={'Seller': 'Seller', 'Count': 'Number of Deals'},
                                        text='Count'
                                    )
                                    fig_seller_cluster_freq.update_traces(textposition='outside')
                                    st.plotly_chart(fig_seller_cluster_freq)

                                    # Monetary value per seller
                                    seller_monetary = cluster_deals.groupby('مسئول معامله')['ارزش معامله'].sum().reset_index()
                                    seller_monetary.columns = ['Seller', 'Monetary']

                                    fig_seller_cluster_monetary = px.bar(
                                        seller_monetary,
                                        x='Seller',
                                        y='Monetary',
                                        title=f"Seller Distribution (Monetary) for Clusters: {', '.join(selected_clusters_seller)}",
                                        labels={'Seller': 'Seller', 'Monetary': 'Total Monetary Value'},
                                        text='Monetary'
                                    )
                                    fig_seller_cluster_monetary.update_traces(textposition='outside')
                                    st.plotly_chart(fig_seller_cluster_monetary)

                                    # ------------------ Deals Table ------------------

                                    st.subheader("Successful Deals")

                                    # Merge with RFM data to get customer details
                                    cluster_deals = cluster_deals.merge(rfm_data[['Customer ID', 'RFM_segment_label', 'VIP Status']], left_on='کد دیدار شخص معامله', right_on='Customer ID', how='left')

                                    deals_table = cluster_deals[['Customer ID', 'نام شخص معامله', 'نام خانوادگی شخص معامله', 'VIP Status', 'RFM_segment_label', 'مسئول معامله', 'تعداد شب ', 'ارزش معامله', 'تاریخ انجام معامله']]

                                    st.write(deals_table)

                                    # Download buttons
                                    csv_data = convert_df(deals_table)
                                    excel_data = convert_df_to_excel(deals_table)

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.download_button(
                                            label="Download data as CSV",
                                            data=csv_data,
                                            file_name='seller_cluster_deals.csv',
                                            mime='text/csv',
                                        )
                                    with col2:
                                        st.download_button(
                                            label="Download data as Excel",
                                            data=excel_data,
                                            file_name='seller_cluster_deals.xlsx',
                                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                        )
                            else:
                                st.warning("Please select at least one VIP status.")
                        else:
                            st.warning("Please select at least one cluster.")

            elif page == 'Sale Channel Analysis':
                # ------------------ Sale Channel Analysis Module ------------------

                st.subheader("Sale Channel Analysis")

                analysis_option_channel = st.radio("Select Analysis Type:", options=['By Sale Channel', 'By Cluster'], key='sale_channel_analysis')

                # VIP Filter for this page
                vip_options_page = sorted(rfm_data['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_channel')

                if select_all_vips_page:
                    selected_vips_channel = vip_options_page
                else:
                    selected_vips_channel = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=[],
                        key='vips_multiselect_channel'
                    )

                if analysis_option_channel == 'By Sale Channel':
                    with st.form(key='sale_channel_form'):
                        selected_sale_channel = st.selectbox("Select a Sale Channel (شیوه آشنایی معامله):", options=sale_channels_options)
                        # Date Range Input
                        min_date = data['تاریخ انجام معامله'].min()
                        max_date = data['تاریخ انجام معامله'].max()

                        # Ensure min_date and max_date are dates
                        if pd.isna(min_date) or pd.isna(max_date):
                            st.warning("Date range is invalid. Please check your data.")
                            return

                        min_date = min_date.date()
                        max_date = max_date.date()

                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key='channel_start_date')
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key='channel_end_date')

                        # Apply Filters button
                        apply_channel_filters = st.form_submit_button(label='Apply Filters')

                    if apply_channel_filters:
                        if selected_sale_channel:
                            if selected_vips_channel:
                                # Filter data based on date range and successful deals
                                date_filtered_data = data[
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date)) &
                                    (data['وضعیت معامله'] == 'موفق')
                                ]

                                # Apply global filters
                                date_filtered_data = date_filtered_data[
                                    (date_filtered_data['عنوان محصول'].isin(selected_products_global)) &
                                    (date_filtered_data['مسئول معامله'].isin(selected_sellers)) &
                                    (date_filtered_data['شیوه آشنایی معامله'].isin(selected_sale_channels)) &
                                    (date_filtered_data['VIP Status'].isin(selected_vips_channel))
                                ]

                                channel_data = date_filtered_data[date_filtered_data['شیوه آشنایی معامله'] == selected_sale_channel]
                                if channel_data.empty:
                                    st.warning("No successful deals found for the selected sale channel and VIP statuses in the specified date range.")
                                else:
                                    # Get customer IDs
                                    channel_customer_ids = channel_data['کد دیدار شخص معامله'].unique()
                                    channel_rfm_data = rfm_data[rfm_data['Customer ID'].isin(channel_customer_ids)]

                                    if channel_rfm_data.empty:
                                        st.warning("No RFM data available for the selected sale channel and VIP statuses.")
                                    else:
                                        # Count of customers in each cluster
                                        cluster_counts_channel = channel_rfm_data['RFM_segment_label'].value_counts().reset_index()
                                        cluster_counts_channel.columns = ['RFM_segment_label', 'Count']

                                        # Plot frequency chart
                                        fig_channel_freq = px.bar(
                                            cluster_counts_channel,
                                            x='RFM_segment_label',
                                            y='Count',
                                            title=f"Cluster Distribution (Frequency) for Sale Channel: {selected_sale_channel}",
                                            labels={'RFM_segment_label': 'RFM Segment', 'Count': 'Number of Customers'},
                                            text='Count',
                                            color='RFM_segment_label',
                                            color_discrete_map=COLOR_MAP
                                        )
                                        fig_channel_freq.update_traces(textposition='outside')
                                        st.plotly_chart(fig_channel_freq)

                                        # Monetary value per cluster
                                        channel_monetary = channel_rfm_data.groupby('RFM_segment_label')['Monetary'].sum().reset_index()
                                        fig_channel_monetary = px.bar(
                                            channel_monetary,
                                            x='RFM_segment_label',
                                            y='Monetary',
                                            title=f"Cluster Distribution (Monetary) for Sale Channel: {selected_sale_channel}",
                                            labels={'RFM_segment_label': 'RFM Segment', 'Monetary': 'Total Monetary Value'},
                                            text='Monetary',
                                            color='RFM_segment_label',
                                            color_discrete_map=COLOR_MAP
                                        )
                                        fig_channel_monetary.update_traces(textposition='outside')
                                        st.plotly_chart(fig_channel_monetary)

                                        # ------------------ Customer Table ------------------

                                        st.subheader("Customer Details")

                                        # Sum of 'تعداد شب ' per customer
                                        customer_nights = channel_data.groupby('کد دیدار شخص معامله')['تعداد شب '].sum().reset_index()
                                        customer_nights.rename(columns={'کد دیدار شخص معامله': 'Customer ID', 'تعداد شب ': 'Total Nights'}, inplace=True)

                                        # Merge with RFM data
                                        customer_details = channel_rfm_data[['Customer ID', 'First Name', 'Last Name', 'VIP Status', 'RFM_segment_label', 'Recency', 'Frequency', 'Monetary']]
                                        customer_details = customer_details.merge(customer_nights, on='Customer ID', how='left').fillna(0)

                                        st.write(customer_details)

                                        # Download buttons
                                        csv_data = convert_df(customer_details)
                                        excel_data = convert_df_to_excel(customer_details)

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.download_button(
                                                label="Download data as CSV",
                                                data=csv_data,
                                                file_name='sale_channel_analysis.csv',
                                                mime='text/csv',
                                            )
                                        with col2:
                                            st.download_button(
                                                label="Download data as Excel",
                                                data=excel_data,
                                                file_name='sale_channel_analysis.xlsx',
                                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                            )
                            else:
                                st.warning("Please select at least one VIP status.")
                        else:
                            st.warning("Please select a sale channel.")

                elif analysis_option_channel == 'By Cluster':
                    # Move checkbox outside the form
                    select_all_clusters_channel = st.checkbox("Select all clusters", value=True, key='select_all_clusters_channel')

                    with st.form(key='sale_channel_cluster_form'):
                        cluster_options = rfm_data['RFM_segment_label'].unique().tolist()
                        cluster_options.sort()

                        if select_all_clusters_channel:
                            selected_clusters_channel = cluster_options
                        else:
                            selected_clusters_channel = st.multiselect(
                                "Select Clusters:",
                                options=cluster_options,
                                default=[],
                                key='clusters_multiselect_channel'
                            )
                        # Date Range Input
                        min_date = data['تاریخ انجام معامله'].min()
                        max_date = data['تاریخ انجام معامله'].max()

                        # Ensure min_date and max_date are dates
                        if pd.isna(min_date) or pd.isna(max_date):
                            st.warning("Date range is invalid. Please check your data.")
                            return

                        min_date = min_date.date()
                        max_date = max_date.date()

                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key='channel_cluster_start_date')
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key='channel_cluster_end_date')

                        # Apply Filters button
                        apply_channel_cluster_filters = st.form_submit_button(label='Apply Filters')

                    if apply_channel_cluster_filters:
                        if selected_clusters_channel:
                            if selected_vips_channel:
                                # Filter data based on date range and successful deals
                                date_filtered_data = data[
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date)) &
                                    (data['وضعیت معامله'] == 'موفق')
                                ]

                                # Apply global filters
                                date_filtered_data = date_filtered_data[
                                    (date_filtered_data['عنوان محصول'].isin(selected_products_global)) &
                                    (date_filtered_data['مسئول معامله'].isin(selected_sellers)) &
                                    (date_filtered_data['شیوه آشنایی معامله'].isin(selected_sale_channels)) &
                                    (date_filtered_data['VIP Status'].isin(selected_vips_channel))
                                ]

                                cluster_customers = rfm_data[rfm_data['RFM_segment_label'].isin(selected_clusters_channel)]['Customer ID'].unique()
                                cluster_deals = date_filtered_data[date_filtered_data['کد دیدار شخص معامله'].isin(cluster_customers)]

                                if cluster_deals.empty:
                                    st.warning("No successful deals found for the selected clusters and VIP statuses in the specified date range.")
                                else:
                                    # Frequency of deals per sale channel
                                    channel_counts = cluster_deals['شیوه آشنایی معامله'].value_counts().reset_index()
                                    channel_counts.columns = ['Sale Channel', 'Count']

                                    fig_channel_cluster_freq = px.bar(
                                        channel_counts,
                                        x='Sale Channel',
                                        y='Count',
                                        title=f"Sale Channel Distribution (Frequency) for Clusters: {', '.join(selected_clusters_channel)}",
                                        labels={'Sale Channel': 'Sale Channel', 'Count': 'Number of Deals'},
                                        text='Count'
                                    )
                                    fig_channel_cluster_freq.update_traces(textposition='outside')
                                    st.plotly_chart(fig_channel_cluster_freq)

                                    # Monetary value per sale channel
                                    channel_monetary = cluster_deals.groupby('شیوه آشنایی معامله')['ارزش معامله'].sum().reset_index()
                                    channel_monetary.columns = ['Sale Channel', 'Monetary']

                                    fig_channel_cluster_monetary = px.bar(
                                        channel_monetary,
                                        x='Sale Channel',
                                        y='Monetary',
                                        title=f"Sale Channel Distribution (Monetary) for Clusters: {', '.join(selected_clusters_channel)}",
                                        labels={'Sale Channel': 'Sale Channel', 'Monetary': 'Total Monetary Value'},
                                        text='Monetary'
                                    )
                                    fig_channel_cluster_monetary.update_traces(textposition='outside')
                                    st.plotly_chart(fig_channel_cluster_monetary)

                                    # ------------------ Deals Table ------------------

                                    st.subheader("Successful Deals")

                                    # Merge with RFM data to get customer details
                                    cluster_deals = cluster_deals.merge(rfm_data[['Customer ID', 'RFM_segment_label', 'VIP Status']], left_on='کد دیدار شخص معامله', right_on='Customer ID', how='left')

                                    deals_table = cluster_deals[['Customer ID', 'نام شخص معامله', 'نام خانوادگی شخص معامله', 'VIP Status', 'RFM_segment_label', 'شیوه آشنایی معامله', 'تعداد شب ', 'ارزش معامله', 'تاریخ انجام معامله']]

                                    st.write(deals_table)

                                    # Download buttons
                                    csv_data = convert_df(deals_table)
                                    excel_data = convert_df_to_excel(deals_table)

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.download_button(
                                            label="Download data as CSV",
                                            data=csv_data,
                                            file_name='sale_channel_cluster_deals.csv',
                                            mime='text/csv',
                                        )
                                    with col2:
                                        st.download_button(
                                            label="Download data as Excel",
                                            data=excel_data,
                                            file_name='sale_channel_cluster_deals.xlsx',
                                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                        )
                            else:
                                st.warning("Please select at least one VIP status.")
                        else:
                            st.warning("Please select at least one cluster.")

            elif page == 'VIP Analysis':
                # ------------------ VIP Analysis ------------------

                st.subheader("VIP Analysis")

                # VIP Filter
                vip_options_page = sorted(rfm_data['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_vip_analysis')

                if select_all_vips_page:
                    selected_vips_vip_analysis = vip_options_page
                else:
                    selected_vips_vip_analysis = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=[],
                        key='vips_multiselect_vip_analysis'
                    )

                # Date Range Input
                min_date = data['تاریخ انجام معامله'].min()
                max_date = data['تاریخ انجام معامله'].max()

                # Ensure min_date and max_date are dates
                if pd.isna(min_date) or pd.isna(max_date):
                    st.warning("Date range is invalid. Please check your data.")
                    return

                min_date = min_date.date()
                max_date = max_date.date()

                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key='vip_analysis_start_date')
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key='vip_analysis_end_date')

                if not selected_vips_vip_analysis:
                    st.warning("Please select at least one VIP status.")
                else:
                    # Filter data based on date range and successful deals
                    date_filtered_data = data[
                        (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                        (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date)) &
                        (data['وضعیت معامله'] == 'موفق')
                    ]

                    # Apply VIP filter
                    date_filtered_data = date_filtered_data[date_filtered_data['VIP Status'].isin(selected_vips_vip_analysis)]

                    if date_filtered_data.empty:
                        st.warning("No successful deals found for the selected VIP statuses in the specified date range.")
                    else:
                        # Get customer IDs
                        vip_customer_ids = date_filtered_data['کد دیدار شخص معامله'].unique()
                        vip_rfm_data = rfm_data[rfm_data['Customer ID'].isin(vip_customer_ids)]

                        if vip_rfm_data.empty:
                            st.warning("No RFM data available for the selected VIP statuses.")
                        else:
                            # Insights
                            total_vip_customers = vip_rfm_data['Customer ID'].nunique()
                            total_champions = vip_rfm_data[vip_rfm_data['RFM_segment_label'] == 'Champions']['Customer ID'].nunique()
                            total_vip_champions = vip_rfm_data[(vip_rfm_data['RFM_segment_label'] == 'Champions')]['Customer ID'].nunique()
                            total_vip_non_champions = total_vip_customers - total_vip_champions

                            st.write(f"**Total VIP Customers:** {total_vip_customers}")
                            st.write(f"**Total VIP Champions:** {total_vip_champions}")
                            st.write(f"**Total VIP Non-Champions:** {total_vip_non_champions}")

                            # Number of Champions who are not VIP
                            total_champions_all = rfm_data[rfm_data['RFM_segment_label'] == 'Champions']['Customer ID'].nunique()
                            champions_not_vip = total_champions_all - total_vip_champions
                            st.write(f"**Number of Champions who are not VIP:** {champions_not_vip}")

                            # Plot distribution of VIPs across segments
                            vip_segment_distribution = vip_rfm_data['RFM_segment_label'].value_counts().reset_index()
                            vip_segment_distribution.columns = ['RFM_segment_label', 'Count']

                            fig_vip_segments = px.pie(
                                vip_segment_distribution,
                                names='RFM_segment_label',
                                values='Count',
                                color='RFM_segment_label',
                                color_discrete_map=COLOR_MAP,
                                hole=0.4,
                                title='VIP Customers Distribution across RFM Segments'
                            )

                            fig_vip_segments.update_traces(textposition='inside', textinfo='percent+label')

                            st.plotly_chart(fig_vip_segments)

                            # Additional Insights
                            st.subheader("Additional Insights")

                            # Average Monetary Value per VIP Level
                            avg_monetary_vip = vip_rfm_data.groupby('VIP Status')['Monetary'].mean().reset_index()
                            fig_avg_monetary = px.bar(
                                avg_monetary_vip,
                                x='VIP Status',
                                y='Monetary',
                                title='Average Monetary Value per VIP Level',
                                labels={'Monetary': 'Average Monetary Value'},
                                text='Monetary'
                            )
                            fig_avg_monetary.update_traces(textposition='outside')
                            st.plotly_chart(fig_avg_monetary)

                            # Recency Distribution
                            fig_recency_vip = px.histogram(
                                vip_rfm_data,
                                x='Recency',
                                nbins=50,
                                title='Recency Distribution for VIP Customers',
                                color='VIP Status',
                                barmode='overlay'
                            )
                            st.plotly_chart(fig_recency_vip)

                            # Frequency Distribution
                            fig_frequency_vip = px.histogram(
                                vip_rfm_data,
                                x='Frequency',
                                nbins=20,
                                title='Frequency Distribution for VIP Customers',
                                color='VIP Status',
                                barmode='overlay'
                            )
                            st.plotly_chart(fig_frequency_vip)

                            # ------------------ Customer Table ------------------

                            st.subheader("VIP Customer Details")

                            st.write(vip_rfm_data[['Customer ID', 'First Name', 'Last Name', 'VIP Status', 'Recency', 'Frequency', 'Monetary', 'RFM_segment_label']])

                            # Download buttons
                            csv_data = convert_df(vip_rfm_data)
                            excel_data = convert_df_to_excel(vip_rfm_data)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="Download data as CSV",
                                    data=csv_data,
                                    file_name='vip_analysis.csv',
                                    mime='text/csv',
                                )
                            with col2:
                                st.download_button(
                                    label="Download data as Excel",
                                    data=excel_data,
                                    file_name='vip_analysis.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                )

            elif page == 'Customer Inquiry Module':
                # ------------------ Customer Inquiry Module ------------------

                st.subheader("Customer Inquiry Module")

                with st.form(key='customer_inquiry_form'):
                    st.write("Enter at least one of the following fields to search for a customer:")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        input_last_name = st.text_input("Last Name")
                    with col2:
                        input_phone_number = st.text_input("Phone Number")
                    with col3:
                        input_customer_id = st.text_input("Customer ID")

                    submit_inquiry = st.form_submit_button(label='Search')

                if submit_inquiry:
                    if not input_last_name and not input_phone_number and not input_customer_id:
                        st.error("Please enter at least one of Last Name, Phone Number, or Customer ID.")
                    else:
                        # Filter rfm_data based on inputs
                        inquiry_results = rfm_data.copy()

                        if input_last_name:
                            inquiry_results = inquiry_results[inquiry_results['Last Name'].str.contains(input_last_name, na=False)]
                        if input_phone_number:
                            inquiry_results = inquiry_results[inquiry_results['Phone Number'].astype(str).str.contains(input_phone_number)]
                        if input_customer_id:
                            inquiry_results = inquiry_results[inquiry_results['Customer ID'].astype(str).str.contains(input_customer_id)]

                        if inquiry_results.empty:
                            st.warning("No customers found matching the given criteria.")
                        else:
                            st.success(f"Found {len(inquiry_results)} customer(s) matching the criteria.")

                            # Display customer information
                            for index, customer in inquiry_results.iterrows():
                                st.markdown("---")
                                st.subheader(f"Customer ID: {customer['Customer ID']}")
                                st.write(f"**Name:** {customer['First Name']} {customer['Last Name']}")
                                st.write(f"**Phone Number:** {customer['Phone Number']}")
                                st.write(f"**VIP Status:** {customer['VIP Status']}")
                                st.write(f"**Recency:** {customer['Recency']} days")
                                st.write(f"**Frequency:** {customer['Frequency']}")
                                st.write(f"**Monetary:** {round(customer['Monetary'], 2)}")
                                st.write(f"**Segment:** {customer['RFM_segment_label']}")

                                # Fetch deal history
                                customer_deals = data[data['کد دیدار شخص معامله'] == customer['Customer ID']]
                                if customer_deals.empty:
                                    st.write("No deal history available.")
                                else:
                                    st.write("**Deal History:**")
                                    deal_history = customer_deals[['تاریخ انجام معامله', 'عنوان محصول', 'ارزش معامله', 'وضعیت معامله']].copy()
                                    # Adjust monetary values for display
                                    deal_history['ارزش معامله'] = deal_history['ارزش معامله'].round(2)
                                    st.dataframe(deal_history)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an Excel file to proceed.")

if __name__ == '__main__':
    main()
