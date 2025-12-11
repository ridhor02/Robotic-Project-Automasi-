import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# === Konfigurasi Dasar ===
st.set_page_config(page_title="Inventory Forecast Tool", layout="wide")
st.title("📦 Inventory Forecast & Optimization Tool (Hybrid + Comparison Mode)")
st.caption("Prediksi kebutuhan, optimasi stok, dan analisis metode terbaik otomatis dari Google Sheets")

# === 1. Ambil Data dari Google Sheets ===
SHEET_URL = "https://docs.google.com/spreadsheets/d/1GIMbsZEdC16m67GY0FG-1tq-hnfs8xMiJZIoTtdEb_M/gviz/tq?tqx=out:csv&sheet=Forecaset"

st.info("📡 Mengambil data otomatis dari Google Sheets...")
try:
    df = pd.read_csv(SHEET_URL)
    required_cols = ['StockCode', 'PartNumber', 'Mnemonic', 'ItemName', 'Month', 'UsageQty', 'LeadTime', 'MOQ']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Kolom wajib belum lengkap. Pastikan ada: {', '.join(required_cols)}")
        st.stop()

    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values(['StockCode', 'Month'])
    
    # Validasi dan cleaning data
    df['UsageQty'] = pd.to_numeric(df['UsageQty'], errors='coerce')
    df['LeadTime'] = pd.to_numeric(df['LeadTime'], errors='coerce')
    df['MOQ'] = pd.to_numeric(df['MOQ'], errors='coerce')
    
    # Filter outlier dan nilai negatif
    df = df[df['UsageQty'] >= 0]
    df = df[df['LeadTime'] >= 0]
    df = df[df['MOQ'] >= 0]
    
    # Fill NaN values
    df['UsageQty'] = df['UsageQty'].fillna(0)
    df['LeadTime'] = df['LeadTime'].fillna(1)
    df['MOQ'] = df['MOQ'].fillna(0)
    
    st.success(f"✅ Data berhasil dimuat! Total {df['StockCode'].nunique()} item.")
    
except Exception as e:
    st.error(f"❌ Gagal mengambil data dari Google Sheets: {e}")
    st.stop()

# === 2. Pengaturan Parameter yang Diperbaiki ===
st.sidebar.header("⚙️ Pengaturan yang Dioptimasi")

# Opsi pemilihan metode forecast
forecast_method = st.sidebar.selectbox(
    "Pilih Metode Forecast:",
    ["Auto (Hybrid)", "Enhanced Prophet", "Seasonal Moving Average", "Trend-Based", "Manual Selection"],
    help="Pilih metode forecasting yang diinginkan"
)

forecast_period = st.sidebar.number_input("Periode Forecast (bulan)", min_value=1, max_value=12, value=3)

# === FITUR BARU: TOGGLE SERVICE LEVEL / Z-FACTOR ===
st.sidebar.subheader("🎯 Pengaturan Service Level & Safety Stock")

use_service_level = st.sidebar.checkbox(
    "Gunakan Service Level & Safety Stock Calculation", 
    value=True,
    help="Aktifkan perhitungan safety stock berdasarkan service level dan z-factor"
)

if use_service_level:
    service_level = st.sidebar.slider("Service Level (%)", 90, 99, 95)
    
    # Fungsi untuk mendapatkan z_value berdasarkan service level
    def get_z_value(service_level):
        z_map = {90: 1.28, 91: 1.34, 92: 1.41, 93: 1.48, 94: 1.55, 95: 1.65, 96: 1.75, 97: 1.88, 98: 2.05, 99: 2.33}
        return z_map.get(service_level, 1.65)
    
    # Dapatkan z_value berdasarkan input user
    current_z_value = get_z_value(service_level)
    
    # Tampilkan z_value saat ini di sidebar
    st.sidebar.info(f"**Z-Value untuk {service_level}%:** {current_z_value}")
else:
    service_level = 95
    current_z_value = 0  # Z-value = 0 berarti tidak ada safety stock
    st.sidebar.info("**Safety Stock Calculation:** ❌ Non-Aktif")

window_ma = st.sidebar.slider("Window Moving Average (bulan)", 2, 12, 6)

# Parameter baru untuk mengontrol forecast
st.sidebar.subheader("🔧 Kontrol Forecast")
smoothing_factor = st.sidebar.slider("Faktor Smoothing", 0.1, 1.0, 0.7, 
                                   help="Mengurangi fluktuasi forecast")
cap_factor = st.sidebar.slider("Faktor Pembatas (Cap)", 1.5, 5.0, 2.5,
                             help="Membatasi forecast maksimum relatif terhadap rata-rata historis")
use_log_transform = st.sidebar.checkbox("Gunakan Transformasi Log", value=True,
                                      help="Stabilkan data dengan transformasi log")

# === FITUR BARU: ADJUST MAX STOCK BERDASARKAN RATIO ===
st.sidebar.subheader("📊 Adjust Max Stock Berdasarkan Ratio")

enable_ratio_adjustment = st.sidebar.checkbox(
    "Aktifkan Adjust Max Berdasarkan Ratio Forecast/Historical", 
    value=True,
    help="Sesuaikan Max Stock berdasarkan perbandingan forecast vs historical average"
)

ratio_adjustment_factor = st.sidebar.slider(
    "Faktor Adjustmen Ratio", 
    min_value=0.5, 
    max_value=3.0, 
    value=1.2,
    help="Faktor pengali untuk menyesuaikan Max Stock berdasarkan ratio forecast/historical"
)

# Threshold untuk adjustment
ratio_threshold_high = st.sidebar.slider(
    "Threshold Ratio Tinggi", 
    min_value=1.5, 
    max_value=3.0, 
    value=2.0,
    help="Ratio di atas ini dianggap kenaikan signifikan"
)

ratio_threshold_low = st.sidebar.slider(
    "Threshold Ratio Rendah", 
    min_value=0.1, 
    max_value=0.8, 
    value=0.5,
    help="Ratio di bawah ini dianggap penurunan signifikan"
)

# === PARAMETER KHUSUS UNTUK ITEM BARU ===
st.sidebar.subheader("🆕 Setting Item Baru & Data Terbatas")

# Metode untuk item baru
new_item_method = st.sidebar.selectbox(
    "Metode Forecast untuk Data Terbatas:",
    ["Berdasarkan Kategori", "Rata-rata Semua Item", "Manual Input", "Moving Average Sederhana"],
    help="Cara menentukan forecast untuk item dengan data historis terbatas"
)

# Default values untuk item baru
default_forecast_new_item = st.sidebar.number_input(
    "Default Forecast (jika manual):", 
    min_value=0, 
    value=100,
    help="Nilai forecast default untuk item dengan data terbatas"
)

# Threshold untuk data terbatas
data_threshold = st.sidebar.slider(
    "Minimal Data untuk Forecasting Normal (bulan):", 
    min_value=2, max_value=6, value=4,
    help="Item dengan data kurang dari ini akan menggunakan metode khusus"
)

# === FUNGSI ANALISIS TREND TAHUNAN ===
def calculate_annual_trend(data):
    """Hitung trend produksi tahunan dengan persentase perubahan"""
    # Ekstrak tahun dari data
    data_copy = data.copy()
    data_copy['Year'] = data_copy['Month'].dt.year
    
    # Group by tahun dan hitung total usage per tahun
    annual_data = data_copy.groupby('Year')['UsageQty'].sum().reset_index()
    annual_data = annual_data.sort_values('Year')
    
    # Hitung persentase perubahan tahunan
    annual_data['YoY_Change'] = annual_data['UsageQty'].pct_change() * 100
    annual_data['YoY_Absolute_Change'] = annual_data['UsageQty'].diff()
    
    # Hitung trend 3 tahun terakhir jika data mencukupi
    if len(annual_data) >= 3:
        recent_3_years = annual_data.tail(3)
        trend_3y = recent_3_years['YoY_Change'].mean()
        volatility_3y = recent_3_years['YoY_Change'].std()
    else:
        trend_3y = annual_data['YoY_Change'].mean() if len(annual_data) > 1 else 0
        volatility_3y = 0
    
    # Klasifikasikan trend
    if trend_3y > 10:
        trend_category = "📈 Pertumbuhan Kuat"
    elif trend_3y > 5:
        trend_category = "📈 Pertumbuhan Sedang"
    elif trend_3y > 0:
        trend_category = "↗️ Pertumbuhan Ringan"
    elif trend_3y == 0:
        trend_category = "➡️ Stagnan"
    elif trend_3y > -5:
        trend_category = "↘️ Penurunan Ringan"
    elif trend_3y > -10:
        trend_category = "📉 Penurunan Sedang"
    else:
        trend_category = "📉 Penurunan Kuat"
    
    return annual_data, trend_3y, trend_category, volatility_3y

def calculate_seasonal_pattern(data):
    """Hitung pola musiman per bulan"""
    data_copy = data.copy()
    data_copy['MonthNumber'] = data_copy['Month'].dt.month
    monthly_pattern = data_copy.groupby('MonthNumber')['UsageQty'].agg(['mean', 'std', 'count']).reset_index()
    monthly_pattern['seasonal_factor'] = monthly_pattern['mean'] / monthly_pattern['mean'].mean()
    
    return monthly_pattern

def analyze_business_impact(trend_analysis, forecast_results):
    """Analisis dampak bisnis dari trend produksi"""
    # Hitung total forecast vs historical
    total_historical = trend_analysis['annual_data']['UsageQty'].sum()
    total_forecast = forecast_results['Forecast_Total_Bulanan'].sum()
    
    # Growth rate forecast
    growth_rate = ((total_forecast / total_historical) - 1) * 100 if total_historical > 0 else 0
    
    # Kategori dampak bisnis
    if growth_rate > 20:
        impact_level = "🚀 Pertumbuhan Eksponensial"
        recommendation = "Tingkatkan kapasitas produksi dan persediaan bahan baku"
    elif growth_rate > 10:
        impact_level = "📈 Pertumbuhan Tinggi"
        recommendation = "Optimasi rantai pasok dan pertimbangkan safety stock tambahan"
    elif growth_rate > 0:
        impact_level = "↗️ Pertumbuhan Stabil"
        recommendation = "Pertahankan level persediaan dengan monitoring ketat"
    elif growth_rate == 0:
        impact_level = "➡️ Stagnan"
        recommendation = "Efisiensi inventory dan reduksi slow-moving items"
    elif growth_rate > -10:
        impact_level = "↘️ Penurunan Ringan"
        recommendation = "Review produk dan kurangi persediaan bertahap"
    else:
        impact_level = "📉 Penurunan Signifikan"
        recommendation = "Fokus pada produk core dan kurangi inventory secara agresif"
    
    impact_analysis = {
        'total_historical': total_historical,
        'total_forecast': total_forecast,
        'growth_rate': growth_rate,
        'impact_level': impact_level,
        'recommendation': recommendation,
        'annual_trend': trend_analysis['trend_3y'],
        'trend_category': trend_analysis['trend_category']
    }
    
    return impact_analysis

# === FUNGSI BARU: VARIASI BULANAN ===
def detect_seasonal_pattern(historical_data):
    """Deteksi pola musiman dari data historis"""
    if len(historical_data) < 6:  # Minimal 6 bulan untuk pattern detection
        return None
    
    monthly_pattern = historical_data.groupby(historical_data['Month'].dt.month)['UsageQty'].mean()
    if monthly_pattern.mean() > 0:
        seasonal_factor = monthly_pattern / monthly_pattern.mean()
        return seasonal_factor
    return None

def enhanced_prophet_forecast(data, item, period, cap_factor=2.5, use_log=False):
    """Enhanced Prophet dengan seasonal detection"""
    item_data = data[data['StockCode'] == item][['Month', 'UsageQty']].copy()
    
    if len(item_data) < 2:
        return None
    
    item_data = item_data.rename(columns={'Month': 'ds', 'UsageQty': 'y'})
    
    # Handle zero and negative values untuk log transform
    if use_log:
        item_data['y'] = np.log1p(item_data['y'])
    
    # Set cap untuk growth limitation
    max_historical = item_data['y'].max()
    item_data['cap'] = max_historical * cap_factor
    
    try:
        model = Prophet(
            growth='linear',
            yearly_seasonality=True if len(item_data) >= 12 else False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=0.1
        )
        
        model.fit(item_data)
        
        future = model.make_future_dataframe(periods=period, freq='M')
        future['cap'] = max_historical * cap_factor
        
        forecast = model.predict(future)
        fc = forecast[['ds', 'yhat']].tail(period)
        
        # Convert back dari log jika menggunakan transformasi
        if use_log:
            fc['yhat'] = np.expm1(fc['yhat'])
        
        # Ensure positive values
        fc['yhat'] = fc['yhat'].apply(lambda x: max(x, 0))
        
        return fc
    except Exception:
        return None

def seasonal_moving_average_forecast(data, item, period, window, smoothing=0.7):
    """Moving Average dengan variasi bulanan yang realistic"""
    item_data = data[data['StockCode'] == item][['Month', 'UsageQty']].copy()
    
    if len(item_data) < max(2, window//2):
        return None
    
    # 1. Hitung base forecast dengan weighted MA
    available_window = min(window, len(item_data))
    weights = np.exp(np.linspace(-1, 0, available_window))
    weights /= weights.sum()
    base_forecast = np.average(item_data['UsageQty'].tail(available_window), weights=weights)
    
    # 2. Hitung historical variation pattern
    historical_std = item_data['UsageQty'].std()
    historical_mean = item_data['UsageQty'].mean()
    historical_cv = historical_std / (historical_mean + 0.001)  # Coefficient of variation
    
    # 3. Deteksi pola seasonal jika data cukup
    seasonal_pattern = detect_seasonal_pattern(item_data)
    
    # 4. Generate forecast dengan variasi realistic
    future_dates = pd.date_range(item_data['Month'].max(), periods=period+1, freq='M')[1:]
    forecast_values = []
    
    for i, date in enumerate(future_dates):
        month = date.month
        
        # Base forecast dengan seasonal adjustment jika ada pattern
        if seasonal_pattern is not None and month in seasonal_pattern.index:
            seasonal_forecast = base_forecast * seasonal_pattern[month]
        else:
            seasonal_forecast = base_forecast
        
        # Add realistic variation berdasarkan historical pattern
        if historical_cv > 0.1 and historical_std > 0:
            variation = np.random.normal(0, historical_std * 0.3)
        else:
            variation = base_forecast * 0.15 * (np.random.random() - 0.5)
        
        monthly_forecast = seasonal_forecast + variation
        monthly_forecast = max(monthly_forecast, 0)
        
        # Smoothing dengan historical average
        final_forecast = smoothing * monthly_forecast + (1 - smoothing) * historical_mean
        forecast_values.append(final_forecast)
    
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    
    # 5. Adjust total agar match dengan base forecast total
    current_total = sum(forecast_values)
    target_total = base_forecast * period
    
    if current_total > 0 and abs(current_total - target_total) / target_total > 0.1:
        adjustment_factor = target_total / current_total
        forecast_df['yhat'] = forecast_df['yhat'] * adjustment_factor
    
    return forecast_df

def trend_based_forecast(data, item, period):
    """Forecast dengan memperhatikan trend untuk data terbatas"""
    item_data = data[data['StockCode'] == item][['Month', 'UsageQty']].copy()
    
    if len(item_data) < 2:
        return None
    
    # Hitung simple trend
    x = np.arange(len(item_data))
    y = item_data['UsageQty'].values
    
    try:
        # Linear regression untuk trend
        slope, intercept = np.polyfit(x, y, 1)
        
        # Buat forecast dengan trend
        future_dates = pd.date_range(item_data['Month'].max(), periods=period+1, freq='M')[1:]
        forecast_values = []
        
        for i in range(period):
            trend_value = intercept + slope * (len(item_data) + i)
            
            # Add variation berdasarkan historical pattern
            if len(item_data) >= 3:
                historical_std = item_data['UsageQty'].std()
                variation = np.random.normal(0, historical_std * 0.4)
            else:
                variation = trend_value * 0.2 * (np.random.random() - 0.5)
            
            final_forecast = max(trend_value + variation, 0)
            forecast_values.append(final_forecast)
        
        return pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    except:
        return None

# === FUNGSI UNTUK ITEM BARU & DATA TERBATAS ===
def handle_new_item(item_data, method="Berdasarkan Kategori", default_value=100, all_items_data=None):
    """Menangani item baru atau item dengan data historis terbatas"""
    if method == "Manual Input":
        forecast_value = default_value
        method_used = "Manual (Data Terbatas)"
        
    elif method == "Rata-rata Semua Item":
        if all_items_data is not None and not all_items_data.empty:
            forecast_value = all_items_data['UsageQty'].mean()
        else:
            forecast_value = default_value
        method_used = "Rata-rata Global (Data Terbatas)"
        
    elif method == "Moving Average Sederhana":
        if len(item_data) > 0:
            forecast_value = item_data['UsageQty'].mean()
        else:
            forecast_value = default_value
        method_used = "MA Sederhana (Data Terbatas)"
        
    else:  # "Berdasarkan Kategori"
        if all_items_data is not None and not all_items_data.empty:
            forecast_value = all_items_data['UsageQty'].median()
        else:
            forecast_value = default_value
        method_used = "Kategori Based (Data Terbatas)"
    
    # Untuk data terbatas, buat forecast dengan variasi
    future_dates = pd.date_range(pd.Timestamp.now(), periods=forecast_period+1, freq='M')[1:]
    forecast_values = []
    
    for i in range(forecast_period):
        variation = forecast_value * 0.2 * (np.random.random() - 0.5)
        monthly_forecast = max(forecast_value + variation, 0)
        forecast_values.append(monthly_forecast)
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values
    })
    
    return forecast_df, method_used, forecast_value

# === FUNGSI BARU: ADJUST MAX STOCK BERDASARKAN RATIO ===
def adjust_max_stock_by_ratio(base_max_stock, ratio_forecast_historical, adjustment_factor=1.2, 
                            threshold_high=2.0, threshold_low=0.5, moq=0):
    """
    Sesuaikan Max Stock berdasarkan ratio forecast vs historical average
    """
    
    # Jika ratio tidak tersedia atau invalid, return base max stock
    if ratio_forecast_historical == 'N/A' or pd.isna(ratio_forecast_historical):
        return base_max_stock, "No Adjustment", "Ratio tidak tersedia"
    
    try:
        ratio = float(ratio_forecast_historical)
    except:
        return base_max_stock, "No Adjustment", "Ratio tidak valid"
    
    # Case 1: Ratio sangat tinggi (forecast >> historical)
    if ratio >= threshold_high:
        adjustment = base_max_stock * adjustment_factor
        adjustment_type = "Increased"
        adjustment_reason = f"Forecast {ratio:.2f}x > historical (threshold: {threshold_high})"
    
    # Case 2: Ratio sangat rendah (forecast << historical)
    elif ratio <= threshold_low:
        adjustment = base_max_stock * (1 / adjustment_factor)
        adjustment_type = "Decreased"
        adjustment_reason = f"Forecast {ratio:.2f}x < historical (threshold: {threshold_low})"
    
    # Case 3: Ratio normal
    else:
        if ratio > 1.0:
            minor_adjustment = 1.0 + (ratio - 1.0) * 0.3
            adjustment = base_max_stock * minor_adjustment
            adjustment_type = "Minor Increase"
            adjustment_reason = f"Forecast {ratio:.2f}x > historical (minor adjustment)"
        elif ratio < 1.0:
            minor_adjustment = 1.0 - (1.0 - ratio) * 0.2
            adjustment = base_max_stock * minor_adjustment
            adjustment_type = "Minor Decrease"
            adjustment_reason = f"Forecast {ratio:.2f}x < historical (minor adjustment)"
        else:
            adjustment = base_max_stock
            adjustment_type = "No Adjustment"
            adjustment_reason = "Forecast ≈ historical"
    
    # Pastikan adjusted max stock memenuhi MOQ requirements
    if moq > 0:
        adjustment = max(adjustment, moq)
        if adjustment > base_max_stock:
            adjustment = ((adjustment + moq - 1) // moq) * moq
        else:
            adjustment = max(adjustment, base_max_stock * 0.8)
    
    # Batasi adjustment agar tidak terlalu ekstrem
    max_allowed_increase = base_max_stock * 2.0
    min_allowed_decrease = base_max_stock * 0.5
    
    adjustment = max(min(adjustment, max_allowed_increase), min_allowed_decrease)
    
    return round(adjustment), adjustment_type, adjustment_reason

# === FUNGSI OPTIMASI STOK YANG SUDAH DIPERBAIKI ===
def optimize_stock(forecasted, lead_time, moq, z_value, historical_data=None, is_new_item=False, method_used=""):
    """Fungsi optimasi stok dengan penyesuaian MOQ yang benar DAN ratio adjustment"""
    try:
        # Validasi input
        if forecasted is None or forecasted.empty:
            return 0, 0, 0, "No Adjustment", "No forecast data", 'N/A'
        
        if pd.isna(lead_time) or lead_time <= 0:
            lead_time = 1
        
        if pd.isna(moq) or moq < 0:
            moq = 0
        
        avg_demand = forecasted['yhat'].mean()
        
        if is_new_item:
            min_s, max_s, ss = optimize_stock_for_new_item(avg_demand, lead_time, moq, z_value, method_used)
            adjustment_type = "New Item Default"
            adjustment_reason = "Item dengan data terbatas"
            ratio_forecast_historical = 'N/A'
            return min_s, max_s, ss, adjustment_type, adjustment_reason, ratio_forecast_historical
        
        # Hitung safety stock
        if historical_data is not None and len(historical_data) > 1:
            std_demand = historical_data.std()
        else:
            std_demand = forecasted['yhat'].std()
        
        # Batasi std_demand agar tidak terlalu ekstrem
        if avg_demand > 0:
            std_demand = min(std_demand, avg_demand * 1.5)
        
        # Hitung safety stock dengan lead time dalam bulan
        lead_time_months = max(lead_time / 30, 0.1)
        
        # Safety stock calculation - TERGANTUNG use_service_level
        if use_service_level and z_value > 0:
            safety_stock = z_value * std_demand * np.sqrt(lead_time_months)
            safety_stock = max(safety_stock, avg_demand * 0.1)
        else:
            safety_stock = avg_demand * 0.1 if avg_demand > 0 else 0
        
        # Hitung reorder point
        reorder_point = avg_demand * lead_time_months + safety_stock
        
        # Hitung Min Stock - SESUAIKAN DENGAN MOQ
        min_stock_calculated = reorder_point
        if moq > 0:
            if moq > min_stock_calculated:
                min_stock = moq
            else:
                min_stock = max(moq, ((min_stock_calculated + moq - 1) // moq) * moq)
        else:
            min_stock = max(min_stock_calculated, avg_demand * 0.5) if avg_demand > 0 else 0
        
        # Hitung Base Max Stock
        if use_service_level and z_value > 0:
            base_max_stock = min_stock + safety_stock + (avg_demand * 1.0)
        else:
            base_max_stock = min_stock + (avg_demand * 1.5) if avg_demand > 0 else min_stock
        
        if moq > 0 and base_max_stock > min_stock:
            stock_range = base_max_stock - min_stock
            adjusted_range = ((stock_range + moq - 1) // moq) * moq
            base_max_stock = min_stock + adjusted_range
        
        base_max_stock = max(base_max_stock, min_stock + moq) if moq > 0 else base_max_stock
        
        # Hitung ratio forecast vs historical
        ratio_forecast_historical = 'N/A'
        if historical_data is not None and len(historical_data) > 0:
            historical_mean = historical_data.mean()
            if historical_mean > 0:
                ratio_forecast_historical = avg_demand / historical_mean
        
        return (round(min_stock), round(base_max_stock), round(safety_stock), 
                "Base Adjustment", "Base calculation", ratio_forecast_historical)
                
    except Exception as e:
        return 0, 0, 0, "Error", str(e), 'N/A'

def optimize_stock_for_new_item(forecast_value, lead_time, moq, z_value, method_used):
    """Optimasi stok khusus untuk item baru atau data terbatas dengan MOQ adjustment"""
    # Validasi input
    if pd.isna(forecast_value) or forecast_value <= 0:
        forecast_value = default_forecast_new_item
    
    if pd.isna(lead_time) or lead_time <= 0:
        lead_time = 1
    
    if pd.isna(moq) or moq < 0:
        moq = 0
    
    # Tentukan std_demand berdasarkan metode
    if "Manual" in method_used:
        std_demand = forecast_value * 0.5
    else:
        std_demand = forecast_value * 0.3
    
    # Hitung safety stock
    lead_time_months = max(lead_time / 30, 0.1)
    
    # Safety stock calculation
    if use_service_level and z_value > 0:
        safety_stock = z_value * std_demand * np.sqrt(lead_time_months)
        safety_stock = min(safety_stock, forecast_value * 2.0)
    else:
        safety_stock = forecast_value * 0.2
    
    # Reorder point
    reorder_point = forecast_value * lead_time_months + safety_stock
    
    # Min Stock - SESUAIKAN DENGAN MOQ
    if moq > 0:
        if moq > reorder_point:
            min_stock = moq
        else:
            min_stock = ((reorder_point + moq - 1) // moq) * moq
    else:
        min_stock = max(reorder_point, forecast_value * 0.5)
    
    # Max Stock
    if use_service_level and z_value > 0:
        max_stock = min_stock + safety_stock + (forecast_value * 1.5)
    else:
        max_stock = min_stock + (forecast_value * 2.0)
    
    # Jika MOQ ada, pastikan perbedaan min dan max adalah kelipatan MOQ yang wajar
    if moq > 0:
        range_stock = max_stock - min_stock
        if range_stock < moq * 2:
            max_stock = min_stock + moq * 2
    
    return round(min_stock), round(max_stock), round(safety_stock)

# === FUNGSI VALIDASI TAMBAHAN UNTUK MEMASTIKAN MOQ ===
def validate_moq_adherence(results_df):
    """Validasi bahwa semua min stock >= MOQ"""
    moq_issues = []
    for idx, row in results_df.iterrows():
        min_stock = row['MinStock'] if 'MinStock' in row else 0
        moq = row['MOQ'] if 'MOQ' in row else 0
        
        if moq > 0 and min_stock < moq:
            moq_issues.append(f"Item {row['StockCode']}: MinStock ({min_stock}) < MOQ ({moq})")
    
    if moq_issues:
        st.warning(f"⚠️ Terdapat {len(moq_issues)} item dengan potensi issue MOQ compliance")
        for issue in moq_issues[:3]:
            st.write(f"  - {issue}")
        if len(moq_issues) > 3:
            st.write(f"  - ... dan {len(moq_issues) - 3} item lainnya")
    
    return results_df

# === FUNGSI FORECAST BERDASARKAN METODE YANG DIPILIH ===
def apply_selected_forecast_method(data, item, period, method, window, smoothing, cap_factor, all_items_data=None):
    """Terapkan metode forecast berdasarkan pilihan user"""
    item_data = data[data['StockCode'] == item]
    data_count = len(item_data)
    
    # Jika tidak ada data sama sekali
    if data_count == 0:
        return None, "Tidak Ada Data", True
    
    # Untuk data terbatas, gunakan metode khusus
    if data_count < data_threshold:
        if method in ["Auto (Hybrid)", "Manual Selection"]:
            trend_fc = trend_based_forecast(data, item, period)
            if trend_fc is not None:
                return trend_fc, "Trend-Based (Data Terbatas)", True
            else:
                fc, method_used, forecast_value = handle_new_item(
                    item_data, new_item_method, default_forecast_new_item, all_items_data
                )
                return fc, method_used, True
        else:
            if method == "Enhanced Prophet":
                fc = enhanced_prophet_forecast(data, item, period, cap_factor, use_log_transform)
                method_used = "Enhanced Prophet (Data Terbatas)"
            elif method == "Seasonal Moving Average":
                fc = seasonal_moving_average_forecast(data, item, period, window, smoothing)
                method_used = "Seasonal MA (Data Terbatas)"
            elif method == "Trend-Based":
                fc = trend_based_forecast(data, item, period)
                method_used = "Trend-Based (Data Terbatas)"
            
            if fc is not None:
                return fc, method_used, True
            else:
                fc, method_used, forecast_value = handle_new_item(
                    item_data, new_item_method, default_forecast_new_item, all_items_data
                )
                return fc, f"Fallback: {method_used}", True
    
    # Untuk data yang cukup
    if method == "Auto (Hybrid)":
        return hybrid_forecast_with_tracking(data, item, period, window, smoothing, cap_factor, all_items_data)
    elif method == "Enhanced Prophet":
        fc = enhanced_prophet_forecast(data, item, period, cap_factor, use_log_transform)
        return fc, "Enhanced Prophet", False if fc is not None else None
    elif method == "Seasonal Moving Average":
        fc = seasonal_moving_average_forecast(data, item, period, window, smoothing)
        return fc, "Seasonal MA", False if fc is not None else None
    elif method == "Trend-Based":
        fc = trend_based_forecast(data, item, period)
        return fc, "Trend-Based", False if fc is not None else None
    elif method == "Manual Selection":
        return hybrid_forecast_with_tracking(data, item, period, window, smoothing, cap_factor, all_items_data)
    
    return None, "Tidak Diketahui", False

def hybrid_forecast_with_tracking(data, item, period, window, smoothing=0.7, cap_factor=2.5, all_items_data=None):
    """Hybrid forecast untuk Auto mode"""
    item_data = data[data['StockCode'] == item]
    data_count = len(item_data)
    
    if data_count < data_threshold:
        trend_fc = trend_based_forecast(data, item, period)
        if trend_fc is not None:
            return trend_fc, "Trend-Based (Data Terbatas)", True
        else:
            fc, method_used, forecast_value = handle_new_item(
                item_data, new_item_method, default_forecast_new_item, all_items_data
            )
            return fc, method_used, True
    else:
        prophet_fc = enhanced_prophet_forecast(data, item, period, cap_factor, use_log_transform)
        seasonal_ma_fc = seasonal_moving_average_forecast(data, item, period, window, smoothing)
        
        if prophet_fc is not None and seasonal_ma_fc is not None:
            prophet_variation = prophet_fc['yhat'].std() / (prophet_fc['yhat'].mean() + 1)
            ma_variation = seasonal_ma_fc['yhat'].std() / (seasonal_ma_fc['yhat'].mean() + 1)
            
            historical_variation = item_data['UsageQty'].std() / (item_data['UsageQty'].mean() + 1)
            
            prophet_diff = abs(prophet_variation - historical_variation)
            ma_diff = abs(ma_variation - historical_variation)
            
            if prophet_diff < ma_diff:
                return prophet_fc, "Enhanced Prophet", False
            else:
                return seasonal_ma_fc, "Seasonal MA", False
                
        elif prophet_fc is not None:
            return prophet_fc, "Enhanced Prophet", False
        elif seasonal_ma_fc is not None:
            return seasonal_ma_fc, "Seasonal MA", False
        else:
            trend_fc = trend_based_forecast(data, item, period)
            if trend_fc is not None:
                return trend_fc, "Trend-Based Fallback", True
            else:
                fc, method_used, forecast_value = handle_new_item(
                    item_data, new_item_method, default_forecast_new_item, all_items_data
                )
                return fc, f"Fallback: {method_used}", True

# === TAMPILKAN ANALISIS TREND TAHUNAN SEBELUM PROSES ===
st.subheader("📊 Analisis Trend Produksi Tahunan")

# Hitung trend tahunan untuk semua data
annual_data, trend_3y, trend_category, volatility_3y = calculate_annual_trend(df)
seasonal_data = calculate_seasonal_pattern(df)

# Tampilkan hasil analisis trend
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Trend 3 Tahun Terakhir", f"{trend_3y:.1f}%")
with col2:
    st.metric("Kategori Trend", trend_category)
with col3:
    st.metric("Volatilitas Trend", f"{volatility_3y:.1f}%")
with col4:
    latest_year = annual_data['Year'].max()
    latest_usage = annual_data[annual_data['Year'] == latest_year]['UsageQty'].iloc[0] if not annual_data[annual_data['Year'] == latest_year].empty else 0
    st.metric(f"Produksi {latest_year}", f"{latest_usage:,.0f}")

# Tampilkan grafik trend tahunan
if len(annual_data) > 1:
    fig_trend, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Trend tahunan
    ax1.plot(annual_data['Year'], annual_data['UsageQty'], marker='o', linewidth=2, markersize=8, color='blue')
    ax1.set_title('Trend Produksi Tahunan', fontweight='bold')
    ax1.set_xlabel('Tahun')
    ax1.set_ylabel('Total Usage Quantity')
    ax1.grid(True, alpha=0.3)

    # Tambahkan anotasi persentase perubahan
    for i, (year, usage, change) in enumerate(zip(annual_data['Year'], annual_data['UsageQty'], annual_data['YoY_Change'])):
        if not pd.isna(change):
            ax1.annotate(f'{change:+.1f}%', (year, usage), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Pola musiman
    ax2.bar(seasonal_data['MonthNumber'], seasonal_data['seasonal_factor'], color='orange', alpha=0.7)
    ax2.set_title('Pola Musiman (Faktor Seasonal)', fontweight='bold')
    ax2.set_xlabel('Bulan')
    ax2.set_ylabel('Faktor Seasonal')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_trend)

# Tampilkan tabel detail trend tahunan
st.subheader("📋 Detail Trend Tahun-ke-Tahun")
trend_display = annual_data.copy()
trend_display['UsageQty'] = trend_display['UsageQty'].apply(lambda x: f"{x:,.0f}")
trend_display['YoY_Change'] = trend_display['YoY_Change'].apply(lambda x: f"{x:+.1f}%" if not pd.isna(x) else "N/A")
trend_display['YoY_Absolute_Change'] = trend_display['YoY_Absolute_Change'].apply(lambda x: f"{x:+,.0f}" if not pd.isna(x) else "N/A")

st.dataframe(trend_display, use_container_width=True)

# === PROSES FORECAST SEMUA ITEM ===
results = []
detailed_forecasts = []
method_usage = {"Enhanced Prophet": 0, "Seasonal MA": 0, "Trend-Based (Data Terbatas)": 0, 
                "Manual (Data Terbatas)": 0, "Rata-rata Global (Data Terbatas)": 0, 
                "MA Sederhana (Data Terbatas)": 0, "Kategori Based (Data Terbatas)": 0,
                "Trend-Based Fallback": 0, "Tidak Cukup Data": 0, "Enhanced Prophet (Data Terbatas)": 0,
                "Seasonal MA (Data Terbatas)": 0}
data_distribution = {1: 0, 2: 0, 3: 0, "4+": 0}

# Tampilkan metode yang dipilih
st.sidebar.success(f"**Metode Forecast:** {forecast_method}")
st.sidebar.success(f"**Service Level:** {'Aktif' if use_service_level else 'Non-Aktif'}")

with st.spinner(f"🔮 Menghitung forecast dengan metode {forecast_method}..."):
    total_items = df['StockCode'].nunique()
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for idx, item in enumerate(df['StockCode'].unique()):
        # Perbaikan progress bar
        progress_value = min((idx + 1) / total_items, 1.0)
        progress_bar.progress(progress_value)
        progress_text.text(f"Memproses item {idx + 1} dari {total_items}")
        
        item_data = df[df['StockCode'] == item]
        item_data_count = len(item_data)
        
        # Skip jika tidak ada data
        if item_data_count == 0:
            method_usage["Tidak Cukup Data"] += 1
            continue
            
        # Distribusi data
        if item_data_count == 1:
            data_distribution[1] += 1
        elif item_data_count == 2:
            data_distribution[2] += 1
        elif item_data_count == 3:
            data_distribution[3] += 1
        else:
            data_distribution["4+"] += 1
        
        # Gunakan metode yang dipilih user
        fc, method_used, is_data_limited = apply_selected_forecast_method(
            df, item, forecast_period, forecast_method, window_ma, 
            smoothing_factor, cap_factor, df
        )
        
        method_usage[method_used] = method_usage.get(method_used, 0) + 1
        
        if fc is None or fc.empty:
            method_usage["Tidak Cukup Data"] += 1
            continue
        
        # Validasi data item
        if len(item_data) == 0:
            method_usage["Tidak Cukup Data"] += 1
            continue
            
        try:
            item_row = item_data.iloc[0]
            lt = max(float(item_row.get('LeadTime', 1)), 1)
            moq = max(float(item_row.get('MOQ', 0)), 0)
        except (IndexError, KeyError) as e:
            method_usage["Tidak Cukup Data"] += 1
            continue
        
        historical_usage = item_data['UsageQty']
        
        # Optimasi stok
        min_s, max_s, ss, adjustment_type, adjustment_reason, ratio_forecast_historical = optimize_stock(
            fc, lt, moq, current_z_value, historical_usage, is_data_limited, method_used
        )
        
        # APPLY RATIO ADJUSTMENT
        if enable_ratio_adjustment and not is_data_limited and ratio_forecast_historical != 'N/A':
            adjusted_max, adj_type, adj_reason = adjust_max_stock_by_ratio(
                max_s, 
                ratio_forecast_historical,
                ratio_adjustment_factor,
                ratio_threshold_high,
                ratio_threshold_low,
                moq
            )
            
            max_s = adjusted_max
            adjustment_type = adj_type
            adjustment_reason = adj_reason
        
        # Validasi MOQ compliance
        if moq > 0:
            if min_s < moq:
                min_s = moq
            if max_s < min_s + moq:
                max_s = min_s + moq
        
        forecast_total = fc['yhat'].sum()
        forecast_avg = fc['yhat'].mean()
        
        # Simpan forecast detail per bulan
        for month_idx, (date, forecast_qty) in enumerate(zip(fc['ds'], fc['yhat'])):
            detailed_forecasts.append({
                'StockCode': item,
                'PartNumber': item_row.get('PartNumber', ''),
                'Mnemonic': item_row.get('Mnemonic', ''),
                'ItemName': item_row.get('ItemName', ''),
                'Bulan_Ke': month_idx + 1,
                'Tanggal': date.strftime('%Y-%m-%d'),
                'Bulan_Tahun': date.strftime('%b %Y'),
                'Forecast_Qty': round(forecast_qty, 2),
                'Metode_Forecast': method_used,
                'Tipe_Item': 'Data Terbatas' if is_data_limited else 'Data Cukup',
                'Jumlah_Data_Historis': len(historical_usage),
                'MOQ': moq,
                'LeadTime': lt
            })
        
        results.append({
            'StockCode': item,
            'PartNumber': item_row.get('PartNumber', ''),
            'Mnemonic': item_row.get('Mnemonic', ''),
            'ItemName': item_row.get('ItemName', ''),
            'Metode_Forecast': method_used,
            'Tipe_Item': 'Data Terbatas' if is_data_limited else 'Data Cukup',
            'Jumlah_Data_Historis': len(historical_usage),
            'Historical_Avg': round(historical_usage.mean(), 2) if len(historical_usage) > 0 else 0,
            'Historical_Max': round(historical_usage.max(), 2) if len(historical_usage) > 0 else 0,
            'Forecast_Per_Bulan': round(forecast_avg, 2),
            'Forecast_Total_Bulanan': round(forecast_total, 2),
            'Periode_Forecast': forecast_period,
            'SafetyStock': ss,
            'MinStock': min_s,
            'MaxStock': max_s,
            'MOQ': moq,
            'LeadTime': lt,
            'ServiceLevel': f"{service_level}%" if use_service_level else "Non-Aktif",
            'Z_Value': current_z_value if use_service_level else "Non-Aktif",
            'Ratio_Forecast_vs_Historical': round(forecast_avg / (historical_usage.mean() + 0.001), 2) if len(historical_usage) > 0 and historical_usage.mean() > 0 else 'N/A',
            'Variasi_Forecast': round(fc['yhat'].std() / (fc['yhat'].mean() + 0.001), 3) if fc['yhat'].mean() > 0 else 0,
            'MOQ_Compliance': '✅' if min_s >= moq else '⚠️',
            'MaxStock_Adjustment': adjustment_type,
            'Adjustment_Reason': adjustment_reason
        })
    
    # Set progress ke 100% setelah selesai
    progress_bar.progress(1.0)
    progress_text.text("✅ Forecast selesai!")

if not results:
    st.warning("Tidak ada data cukup untuk forecasting.")
    st.stop()

output = pd.DataFrame(results)
detailed_forecast_df = pd.DataFrame(detailed_forecasts)

# === ANALISIS DAMPAK BISNIS SETELAH FORECAST ===
st.subheader("💼 Analisis Dampak Bisnis & Rekomendasi Strategis")

# Hitung analisis dampak bisnis
impact_analysis = analyze_business_impact(
    {
        'annual_data': annual_data,
        'trend_3y': trend_3y,
        'trend_category': trend_category,
        'volatility_3y': volatility_3y
    },
    output
)

# Tampilkan hasil analisis dampak bisnis
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Historis", f"{impact_analysis['total_historical']:,.0f}")
with col2:
    st.metric("Total Forecast", f"{impact_analysis['total_forecast']:,.0f}")
with col3:
    st.metric("Growth Rate Forecast", f"{impact_analysis['growth_rate']:.1f}%")
with col4:
    st.metric("Dampak Bisnis", impact_analysis['impact_level'])

# Tampilkan rekomendasi strategis
st.info(f"**📋 Rekomendasi Strategis:** {impact_analysis['recommendation']}")

# === TAMBAHKAN VALIDASI FINAL ===
output = validate_moq_adherence(output)

# === TAMPILKAN HASIL ===
st.subheader("📈 Summary Statistik Forecast & Distribusi Data")

st.info(f"🎯 **Metode Forecast yang Digunakan:** {forecast_method}")
st.info(f"🛡️ **Service Level Calculation:** {'✅ Aktif' if use_service_level else '❌ Non-Aktif'}")
st.info(f"📊 **Analisis Trend Tahunan:** {trend_category} ({trend_3y:.1f}%)")

# Tampilkan distribusi data
st.info(f"📊 **Distribusi Data Historis**: "
        f"1 bulan: {data_distribution[1]} item, "
        f"2 bulan: {data_distribution[2]} item, "
        f"3 bulan: {data_distribution[3]} item, "
        f"4+ bulan: {data_distribution['4+']} item")

# Hitung total forecast
total_forecast_all_items = output['Forecast_Total_Bulanan'].sum()
total_forecast_per_month = output['Forecast_Per_Bulan'].sum()

# Tampilkan statistik penggunaan metode
st.subheader("📋 Distribusi Metode yang Digunakan")

# Filter hanya metode yang digunakan
used_methods = {k: v for k, v in method_usage.items() if v > 0}
cols_count = min(6, len(used_methods))
cols = st.columns(cols_count)

for idx, (method, count) in enumerate(used_methods.items()):
    if idx < cols_count:
        with cols[idx]:
            st.metric(method, f"{count} items")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Forecast/Bulan", f"{total_forecast_per_month:,.0f}")
with col2:
    st.metric(f"Total Forecast/{forecast_period} Bulan", f"{total_forecast_all_items:,.0f}")

# Informasi tentang variasi forecast
avg_variation = output['Variasi_Forecast'].mean()
st.info(f"🔍 **Rata-rata Variasi Forecast**: {avg_variation:.3f} (Coefficient of Variation)")

# === TAMBAHKAN VISUALISASI RATIO ADJUSTMENT ===
if enable_ratio_adjustment:
    st.subheader("📈 Analisis Ratio Forecast vs Historical")
    
    # Filter items yang memiliki ratio valid
    valid_ratio_items = output[output['Ratio_Forecast_vs_Historical'] != 'N/A']
    
    if not valid_ratio_items.empty:
        # Hitung statistik adjustment
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            increased_count = len(valid_ratio_items[valid_ratio_items['MaxStock_Adjustment'].str.contains('Increase')])
            st.metric("Item dengan Max Stock ↑", f"{increased_count}")
        
        with col2:
            decreased_count = len(valid_ratio_items[valid_ratio_items['MaxStock_Adjustment'].str.contains('Decrease')])
            st.metric("Item dengan Max Stock ↓", f"{decreased_count}")
        
        with col3:
            no_adjust_count = len(valid_ratio_items[valid_ratio_items['MaxStock_Adjustment'] == 'No Adjustment'])
            st.metric("Tidak Diadjust", f"{no_adjust_count}")
        
        with col4:
            avg_ratio = valid_ratio_items['Ratio_Forecast_vs_Historical'].mean()
            st.metric("Rata-rata Ratio", f"{avg_ratio:.2f}")

# === TAMPILKAN ITEM BERDASARKAN KATEGORI DATA ===
limited_data_items = output[output['Tipe_Item'] == 'Data Terbatas']
sufficient_data_items = output[output['Tipe_Item'] == 'Data Cukup']

if not limited_data_items.empty:
    st.subheader("🆕 Item dengan Data Terbatas (<4 Bulan)")
    st.dataframe(limited_data_items[[
        'StockCode', 'PartNumber', 'ItemName', 'Metode_Forecast', 'Jumlah_Data_Historis',
        'Forecast_Per_Bulan', 'Forecast_Total_Bulanan', 'Variasi_Forecast',
        'SafetyStock', 'MinStock', 'MaxStock', 'MOQ_Compliance'
    ]], use_container_width=True)

# Tampilkan item dengan data cukup
st.subheader("📦 Hasil Forecast Item dengan Data Cukup (≥4 Bulan)")
display_columns = [
    'StockCode', 'PartNumber', 'Mnemonic', 'ItemName', 'Metode_Forecast', 'Jumlah_Data_Historis',
    'Historical_Avg', 'Forecast_Per_Bulan', 'Forecast_Total_Bulanan', 'Variasi_Forecast',
    'SafetyStock', 'MinStock', 'MaxStock', 'MOQ_Compliance', 'ServiceLevel'
]

if enable_ratio_adjustment:
    display_columns.extend(['MaxStock_Adjustment', 'Adjustment_Reason'])

if not sufficient_data_items.empty:
    st.dataframe(sufficient_data_items[display_columns], use_container_width=True)

# === VISUALISASI DISTRIBUSI METODE ===
st.subheader("📊 Visualisasi Distribusi Metode Forecast")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Pie chart distribusi metode
methods_for_viz = {k: v for k, v in method_usage.items() if v > 0}
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#cc99ff', '#99ccff', '#c2c2f0', '#ffb3e6']
if methods_for_viz:
    ax1.pie(methods_for_viz.values(), labels=methods_for_viz.keys(), autopct='%1.1f%%', 
            colors=colors[:len(methods_for_viz)], startangle=90)
    ax1.set_title('Distribusi Metode Forecast yang Digunakan')
else:
    ax1.text(0.5, 0.5, 'Tidak ada data metode', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Distribusi Metode Forecast yang Digunakan')

# Plot 2: Bar chart variasi per metode
method_variation_data = []
method_labels = []
for method in used_methods.keys():
    if "Tidak Cukup Data" not in method:
        method_items = output[output['Metode_Forecast'] == method]
        if len(method_items) > 0:
            avg_variation = method_items['Variasi_Forecast'].mean()
            method_variation_data.append(avg_variation)
            method_labels.append(method)

if method_variation_data:
    bars = ax2.bar(method_labels, method_variation_data, color=colors[:len(method_labels)], alpha=0.7)
    ax2.set_ylabel('Rata-rata Coefficient of Variation')
    ax2.set_title('Variasi Forecast per Metode')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Tambahkan nilai di atas bar
    for bar, v in zip(bars, method_variation_data):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
else:
    ax2.text(0.5, 0.5, 'Tidak ada data variasi\nuntuk perbandingan', 
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Variasi Forecast per Metode')

plt.tight_layout()
st.pyplot(fig1)

# === DOWNLOAD EXCEL LENGKAP DENGAN GRAFIK ===
st.subheader("💾 Download Hasil Lengkap")

# Buat summary report dengan grafik
col1, col2 = st.columns(2)

with col1:
    # Download Excel dengan semua data
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Sheet utama
        output.to_excel(writer, index=False, sheet_name='Forecast_Summary')
        
        # Sheet analisis trend tahunan
        annual_data.to_excel(writer, index=False, sheet_name='Analisis_Trend_Tahunan')
        
        # Sheet forecast detail per bulan
        detailed_forecast_df.to_excel(writer, index=False, sheet_name='Forecast_Detail_Per_Bulan')
        
        # Sheet statistik metode
        method_stats = pd.DataFrame({
            'Metode': list(method_usage.keys()),
            'Jumlah_Item': list(method_usage.values())
        })
        method_stats.to_excel(writer, index=False, sheet_name='Statistik_Metode')
        
        # Sheet parameter setting
        param_stats = pd.DataFrame({
            'Parameter': ['Metode Forecast', 'Periode Forecast', 'Service Level', 'Z-Value', 'Window MA', 
                         'Smoothing Factor', 'Cap Factor', 'Data Threshold', 'Use Service Level', 
                         'Ratio Adjustment', 'Trend 3 Tahun', 'Trend Kategori'],
            'Nilai': [forecast_method, forecast_period, f"{service_level}%", current_z_value, window_ma, 
                     smoothing_factor, cap_factor, data_threshold, use_service_level, 
                     enable_ratio_adjustment, f"{trend_3y:.1f}%", trend_category]
        })
        param_stats.to_excel(writer, index=False, sheet_name='Parameter_Setting')

    st.download_button(
        label="📊 Download Excel Lengkap",
        data=buffer,
        file_name=f"forecast_{forecast_method}_{forecast_period}bulan.xlsx",
        mime="application/vnd.ms-excel"
    )

st.success(f"""
✅ Forecast berhasil dihitung! 

**📊 Ringkasan Hasil:**
- **Metode**: {forecast_method}
- **Periode**: {forecast_period} bulan kedepan
- **Total Items**: {len(output)} items
- **Data Cukup**: {len(sufficient_data_items)} items, **Data Terbatas**: {len(limited_data_items)} items
- **Service Level**: {'✅ Aktif' if use_service_level else '❌ Non-Aktif'}
- **Ratio Adjustment**: {'✅ Aktif' if enable_ratio_adjustment else '❌ Non-Aktif'}
- **Trend Tahunan**: {trend_category} ({trend_3y:.1f}%)
- **Growth Forecast**: {impact_analysis['growth_rate']:.1f}%
""")
