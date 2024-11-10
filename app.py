import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from datetime import datetime
import jdatetime

# --------------- Helper Functions ---------------

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

# Load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    try:
        data = pd.read_excel(uploaded_file)
        date_columns = [
            'تاریخ انجام معامله', 'تاریخ ایجاد معامله', 'تاریخ احتمالی انجام معامله',
            'تاریخ ورود', 'تاریخ خروج', 'شروع قرارداد', 'پایان قرارداد'
        ]
        for col in date_columns:
            if col in data.columns:
                data[col] = jalali_to_gregorian_vectorized(data[col])
                data[col] = pd.to_datetime(data[col], errors='coerce')
            else:
                st.warning(f"Column '{col}' not found in the data.")

        if 'تعداد شب ' in data.columns:
            data['تعداد شب '] = data['تعداد شب '].astype(str).str.replace(r'\D', '', regex=True)
            data['تعداد شب '] = pd.to_numeric(data['تعداد شب '], errors='coerce')
            data.loc[data['تعداد شب '] > 365, 'تعداد شب '] = np.nan
        else:
            st.warning("Column 'تعداد شب ' not found in the data.")

        if 'ارزش معامله' in data.columns:
            data['ارزش معامله'] = pd.to_numeric(data['ارزش معامله'], errors='coerce')
        else:
            st.warning("Column 'ارزش معامله' not found in the data.")

        data['VIP Status'] = extract_vip_status(data['نام خانوادگی شخص معامله'])
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_rfm(data, today=None):
    data['ارزش معامله'] = data['ارزش معامله'] / 10
    successful_deals = data[data['وضعیت معامله'] == 'موفق']
    if today is None:
        today = datetime.today()
    else:
        today = pd.to_datetime(today)

    rfm_data = successful_deals.groupby('کد دیدار شخص معامله').agg({
        'نام شخص معامله': 'first',
        'نام خانوادگی شخص معامله': 'first',
        'موبایل شخص معامله': 'first',
        'تاریخ انجام معامله': lambda x: (today - pd.to_datetime(x).max()).days,
        'کد دیدار معامله': 'count',
        'ارزش معامله': 'sum',
        'VIP Status': 'first'
    }).reset_index()

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

@st.cache_data
def rfm_segmentation(data):
    data = data[(data['Monetary'] > 0) & (data['Customer ID'] != 0)]
    buckets = data[['Recency', 'Frequency', 'Monetary']].quantile([1/3, 2/3]).to_dict()

    def rfm_segment(row):
        r_score = 1 if row['Recency'] >= 296 else 2 if row['Recency'] >= 185 else 3 if row['Recency'] >= 76 else 4
        f_score = 1 if row['Frequency'] <= buckets['Frequency'][1/3] else 2 if row['Frequency'] <= buckets['Frequency'][2/3] else 3
        m_score = 1 if row['Monetary'] <= buckets['Monetary'][1/3] else 2 if row['Monetary'] <= buckets['Monetary'][2/3] else 3
        return f"{r_score}{f_score}{m_score}"

    data['RFM_segment'] = data.apply(rfm_segment, axis=1)
    segment_labels = {
        '111': 'Churned', '112': 'Churned', '113': 'Lost Big Spenders', '121': 'Churned', '122': 'Churned',
        '123': 'Lost Big Spenders', '131': 'Hibernating', '132': 'Big Loss', '133': 'Big Loss', '211': 'Low Value',
        '212': 'At Risk', '213': 'At Risk', '221': 'Low Value', '222': 'At Risk', '223': 'At Risk', '231': 'At Risk',
        '232': 'At Risk', '233': 'At Risk', '311': 'Low Value', '312': 'Promising', '313': 'Big Spenders',
        '321': 'Promising', '322': 'Promising', '323': 'Promising', '331': 'Loyal Customers', '332': 'Loyal Customers',
        '333': 'Loyal Customers', '411': 'Promising', '412': 'Promising', '413': 'Big Spenders', '421': 'Price Sensitive',
        '422': 'Loyal Customers', '423': 'Loyal Customers', '431': 'Price Sensitive', '432': 'Loyal Customers',
        '433': 'Champions'
    }
    data['RFM_segment_label'] = data['RFM_segment'].map(segment_labels)
    return data

@st.cache_data
def normalize_rfm(data):
    scaler = MinMaxScaler()
    data['Recency_norm'] = scaler.fit_transform(data[['Recency']])
    data['Recency_norm'] = 1 - data['Recency_norm']
    data[['Frequency_norm', 'Monetary_norm']] = scaler.fit_transform(data[['Frequency', 'Monetary']])
    return data

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --------------- Main App ---------------

def main():
    st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")
    st.title("Customer Segmentation Dashboard - Tehran Moble")

    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file:
        data_load_state = st.text('Loading and processing data...')
        data = load_data(uploaded_file)

        if data.empty:
            st.warning("No data loaded. Please check the file and try again.")
            st.stop()

        product_options = sorted(data['عنوان محصول'].dropna().unique().tolist())
        sellers_options = sorted(data['مسئول معامله'].dropna().unique().tolist())
        sale_channels_options = sorted(data['شیوه آشنایی معامله'].dropna().unique().tolist())
        vip_options = sorted(data['VIP Status'].dropna().unique().tolist())

        # Sidebar Navigation
        page = st.sidebar.radio("Go to", [
            'General', 'Compare RFM Segments Over Time', 'Portfolio Analysis', 'Seller Analysis',
            'Sale Channel Analysis', 'VIP Analysis', 'Customer Inquiry Module'
        ])

        # Global Filters
        st.sidebar.header("Global Filters")
        selected_vips = st.sidebar.multiselect("Select VIP Status", options=vip_options, default=vip_options)
        selected_products_global = st.sidebar.multiselect("Select Products", options=product_options, default=product_options)
        selected_sellers = st.sidebar.multiselect("Select Sellers", options=sellers_options, default=sellers_options)
        selected_sale_channels = st.sidebar.multiselect("Select Sale Channels", options=sale_channels_options, default=sale_channels_options)

        # Apply Global Filters
        filtered_data = data[data['VIP Status'].isin(selected_vips) & data['عنوان محصول'].isin(selected_products_global) &
                             data['مسئول معامله'].isin(selected_sellers) & data['شیوه آشنایی معامله'].isin(selected_sale_channels)]

        if filtered_data.empty:
            st.warning("No data available after applying the global filters.")
            st.stop()

        rfm_data = calculate_rfm(data)
        rfm_data = rfm_segmentation(rfm_data)
        rfm_data = normalize_rfm(rfm_data)
        data_load_state.text('Loading and processing data...done!')

        # Implement content for each page like 'General', 'Compare RFM Segments Over Time', etc.

if __name__ == '__main__':
    main()
