# dashboard_navbar_lilac.py
"""
Final Bike Sharing Dashboard
- Horizontal navbar (streamlit-option-menu) with lilac color
- Keeps original functions + adds interactive filters, visualizations, and robust cnt handling
- Random Forest models: rf_day_model.pkl, rf_hour_model.pkl (fallback to data-based estimates if missing)
- Dataset loaded from GitHub (main_data.csv)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# third-party menu
try:
    from streamlit_option_menu import option_menu
except Exception:
    # user might not have it; show helpful message
    st.error("Please install streamlit-option-menu: pip install streamlit-option-menu")
    raise

# -------------------------
# Page config & basic CSS
# -------------------------
st.set_page_config(page_title="Bike Sharing Prediction Dashboard", page_icon="🚲", layout="wide")

st.markdown("""
<style>
/* Header & nav styling */
.header-row { display:flex; align-items:center; gap:12px; }
.main-title { font-size:28px; color:#5b3b8a; font-weight:700; margin-bottom:4px; }
.subtitle { color:#6c757d; margin-top:0; margin-bottom:8px; }

/* Card styles */
.metric-card { background-color:#ffffff; padding:12px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.06); }
.prediction-card { background-color:#fbf7ff; padding:14px; border-left:6px solid #caa3ff; border-radius:8px; }

/* small layout tweaks */
.small-muted { color:#6c757d; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Constants
# -------------------------
DATA_URL = "https://raw.githubusercontent.com/inayahayudeswita/bike-sharing-dashboard/main/data/main_data.csv"

# -------------------------
# Helpers
# -------------------------
def safe_load_csv(url):
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, str(e)

def try_load_model(path):
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        return None, str(e)

def detect_rental_column(df):
    """Detect rental count column. Prefer 'cnt' if present; otherwise try others or create total from casual+registered."""
    if df is None:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    # prefer exact 'cnt'
    if 'cnt' in cols_lower:
        return cols_lower['cnt']
    # common alternatives
    candidates = ['count', 'total', 'rentals', 'rental_count', 'rides', 'rented']
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # casual + registered
    if 'casual' in cols_lower and 'registered' in cols_lower:
        if 'total_rentals' not in df.columns:
            try:
                df['total_rentals'] = pd.to_numeric(df[cols_lower['casual']], errors='coerce').fillna(0) + pd.to_numeric(df[cols_lower['registered']], errors='coerce').fillna(0)
                return 'total_rentals'
            except Exception:
                pass
    # fuzzy search for substrings
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in ['cnt','count','rent','total','rental','ride','trip']):
            return c
    return None

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------------
# Dashboard class (preserve original + enhancements)
# -------------------------
class BikeSharingDashboard:
    def __init__(self):
        # preserve original attributes
        self.day_pipeline = None
        self.hour_pipeline = None
        self.data = None
        self.rental_col = None
        # load models & data
        self.load_models_nonfatal()
        self.load_data()
        self.rental_col = detect_rental_column(self.data)
        self.filters_meta = self.prepare_filters_meta()

    # load models but do not stop app if missing
    def load_models_nonfatal(self):
        day_m, day_err = try_load_model('rf_day_model.pkl')
        hour_m, hour_err = try_load_model('rf_hour_model.pkl')
        self.day_pipeline = day_m
        self.hour_pipeline = hour_m
        # sidebar statuses will show later

    # original load_data but with protections
    def load_data(self):
        df, err = safe_load_csv(DATA_URL)
        if df is None:
            st.sidebar.error(f"Failed to load dataset: {err}")
            self.data = None
            return
        # try parse dates
        if 'dteday' in df.columns:
            try:
                df['dteday'] = pd.to_datetime(df['dteday'])
            except Exception:
                pass
        # coerce numeric for common columns
        numeric_cols = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed']
        for nc in numeric_cols:
            if nc in df.columns:
                df[nc] = pd.to_numeric(df[nc], errors='coerce')
        self.data = df

    # prepare options for filters (meta)
    def prepare_filters_meta(self):
        meta = {}
        if self.data is None:
            return meta
        for c in ['yr','mnth','season','weathersit','weekday','hr']:
            if c in self.data.columns:
                meta[c] = sorted(self.data[c].dropna().unique().tolist())
        return meta

    # original expected features
    def get_expected_features(self, dataset_type):
        if dataset_type == 'day':
            return ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                    'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        else:
            return ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                    'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

    # preserve original preprocess_input but fill defaults if missing
    def preprocess_input(self, input_data, dataset_type):
        expected_features = self.get_expected_features(dataset_type)
        for f in expected_features:
            if f not in input_data:
                if f == 'yr':
                    input_data[f] = datetime.now().year
                elif f == 'mnth':
                    input_data[f] = 1
                elif f == 'hr':
                    input_data[f] = 12
                elif f in ['temp','atemp','hum','windspeed']:
                    input_data[f] = 0.5
                else:
                    input_data[f] = 0
        df = pd.DataFrame([input_data])[expected_features]
        return df, input_data

    # original make_prediction with fallback
    def make_prediction(self, input_df, dataset_type):
        try:
            if dataset_type == 'day':
                if self.day_pipeline is not None:
                    return float(self.day_pipeline.predict(input_df)[0])
            else:
                if self.hour_pipeline is not None:
                    return float(self.hour_pipeline.predict(input_df)[0])
        except Exception as e:
            st.sidebar.error(f"Model prediction error: {e}")
        return float(self.fallback_prediction(input_df, dataset_type))

    def fallback_prediction(self, input_df, dataset_type):
        if self.data is None or self.rental_col is None:
            return 0.0
        df = self.data
        cnt = self.rental_col
        try:
            if dataset_type == 'day' and 'mnth' in input_df.columns and 'mnth' in df.columns:
                m = int(input_df.loc[0,'mnth'])
                grp = df.groupby('mnth')[cnt].mean()
                if m in grp.index:
                    return float(grp.loc[m])
            if dataset_type == 'hour' and 'hr' in input_df.columns and 'hr' in df.columns:
                h = int(input_df.loc[0,'hr'])
                grp = df.groupby('hr')[cnt].mean()
                if h in grp.index:
                    return float(grp.loc[h])
        except Exception:
            pass
        try:
            return float(df[cnt].mean())
        except Exception:
            return 0.0

    # -------------------------
    # UI components
    # -------------------------
    def top_navbar(self):
        """Horizontal navbar using streamlit-option-menu, styled lilac/light purple."""
        selected = option_menu(
            menu_title=None,
            options=["Ringkasan", "Visualisasi", "Prediksi", "Tentang"],
            icons=["list-task", "bar-chart-line", "robot", "info-circle"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#faf5ff"},
                "icon": {"color": "#6f42c1", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px 8px", "--hover-color": "#f3e9ff"},
                "nav-link-selected": {"background-color": "#e9d9ff"},
            },
        )
        return selected

    # preserve original create_input_sidebar but we'll call it in right place
    def create_input_sidebar(self):
        st.sidebar.header("Input Parameters")

        # Dataset type
        dataset_type = st.sidebar.radio("Prediction Type", ['Daily', 'Hourly'], index=0)
        model_type = 'day' if dataset_type == 'Daily' else 'hour'

        # Time features
        col1, col2 = st.sidebar.columns(2)
        with col1:
            season = st.selectbox("Season", [1, 2, 3, 4], index=0, format_func=lambda x: ['Spring','Summer','Fall','Winter'][x-1])
            # year choose from meta if available
            years_meta = self.filters_meta.get('yr', None)
            if years_meta:
                year = st.selectbox("Year", options=years_meta, index=len(years_meta)-1)
            else:
                year = st.number_input("Year (free input for simulation)", min_value=2011, max_value=2100, value=2025, step=1)
            month = st.slider("Month", 1, 12, 6)
        with col2:
            holiday = st.selectbox("Holiday", [0,1], index=0, format_func=lambda x: 'No' if x==0 else 'Yes')
            weekday = st.selectbox("Weekday", [0,1,2,3,4,5,6], index=0, format_func=lambda x: ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][x])
            workingday = st.selectbox("Working Day", [0,1], index=0, format_func=lambda x: 'No' if x==0 else 'Yes')

        hour = 12
        if dataset_type == 'Hourly':
            hr_meta = self.filters_meta.get('hr', None)
            if hr_meta:
                hour = st.selectbox("Hour of Day", options=hr_meta, index=hr_meta.index(12) if 12 in hr_meta else 0)
            else:
                hour = st.slider("Hour of Day", 0, 23, 12)

        st.sidebar.subheader("Weather Conditions")

        weather = st.sidebar.selectbox("Weather Situation", [1,2,3,4], index=0,
                                       format_func=lambda x: ['Clear/Partly Cloudy','Misty/Cloudy','Light Snow/Rain','Heavy Rain/Snow'][x-1])

        temp = st.sidebar.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
        atemp = st.sidebar.slider("Feels Like (normalized)", 0.0, 1.0, 0.5)
        humidity = st.sidebar.slider("Humidity (normalized)", 0.0, 1.0, 0.5)
        windspeed = st.sidebar.slider("Windspeed (normalized)", 0.0, 1.0, 0.1)

        # Prepare input data
        input_data = {
            'season': season,
            'yr': year,
            'mnth': month,
            'holiday': holiday,
            'weekday': weekday,
            'workingday': workingday,
            'weathersit': weather,
            'temp': temp,
            'atemp': atemp,
            'hum': humidity,
            'windspeed': windspeed
        }
        if dataset_type == 'Hourly':
            input_data['hr'] = hour

        # Model status
        st.sidebar.markdown("---")
        st.sidebar.markdown("Model status:")
        st.sidebar.write(f"Day model: {'loaded' if self.day_pipeline is not None else 'not loaded'}")
        st.sidebar.write(f"Hour model: {'loaded' if self.hour_pipeline is not None else 'not loaded'}")

        return input_data, model_type

    # Filters UI for Ringkasan
    def dataset_filters_panel(self):
        if self.data is None:
            st.info("No dataset loaded.")
            return {}
        st.markdown("#### Filters")
        df = self.data
        filters = {}
        # Year range
        if 'yr' in df.columns:
            years = sorted(df['yr'].dropna().unique())
            if len(years) >= 2:
                sel = st.slider("Year range", int(min(years)), int(max(years)), (int(min(years)), int(max(years))))
                filters['yr'] = sel
        # Month multiselect
        if 'mnth' in df.columns:
            months = sorted(df['mnth'].dropna().unique())
            selm = st.multiselect("Months", options=months, default=months)
            filters['mnth'] = selm
        # Season multi
        if 'season' in df.columns:
            seasons = sorted(df['season'].dropna().unique())
            sels = st.multiselect("Seasons", options=seasons, default=seasons)
            filters['season'] = sels
        # Weather
        if 'weathersit' in df.columns:
            weas = sorted(df['weathersit'].dropna().unique())
            selw = st.multiselect("Weather", options=weas, default=weas)
            filters['weathersit'] = selw
        # Hour range
        if 'hr' in df.columns:
            hrs = sorted(df['hr'].dropna().unique())
            if len(hrs) >= 2:
                selh = st.slider("Hour range", int(min(hrs)), int(max(hrs)), (int(min(hrs)), int(max(hrs))))
                filters['hr'] = selh
        # text search
        text = st.text_input("Search (any column)")
        if text:
            filters['text'] = text.strip()
        return filters

    def apply_filters(self, df, filters):
        if df is None or not filters:
            return df
        dff = df.copy()
        for k, v in filters.items():
            if k == 'text' and v:
                mask = pd.Series(False, index=dff.index)
                for c in dff.columns:
                    try:
                        mask = mask | dff[c].astype(str).str.contains(v, case=False, na=False)
                    except Exception:
                        continue
                dff = dff[mask]
            elif k == 'yr' and isinstance(v, tuple):
                dff = dff[(dff['yr'] >= v[0]) & (dff['yr'] <= v[1])]
            elif k == 'mnth' and isinstance(v, list) and v:
                dff = dff[dff['mnth'].isin(v)]
            elif k == 'season' and isinstance(v, list) and v:
                dff = dff[dff['season'].isin(v)]
            elif k == 'weathersit' and isinstance(v, list) and v:
                dff = dff[dff['weathersit'].isin(v)]
            elif k == 'hr' and isinstance(v, tuple):
                dff = dff[(dff['hr'] >= v[0]) & (dff['hr'] <= v[1])]
        return dff

    # show data summary (keeps original content + added interactivity)
    def show_data_summary(self):
        st.header("Ringkasan Data Utama")
        if self.data is None:
            st.info("Dataset not loaded.")
            return

        df = self.data.copy()
        cnt = self.rental_col

        total_records = len(df)
        unique_days = df['dteday'].nunique() if 'dteday' in df.columns else '-'
        unique_hours = df['hr'].nunique() if 'hr' in df.columns else '-'
        mean_cnt = f"{df[cnt].mean():.2f}" if cnt in df.columns else "-"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{total_records:,}")
        c2.metric("Unique Days", unique_days)
        c3.metric("Unique Hours", unique_hours)
        c4.metric("Mean Rentals (overall)", mean_cnt)

        st.markdown("---")
        # Filters + filtered preview
        filters = self.dataset_filters_panel()
        filtered = self.apply_filters(df, filters)

        st.subheader("Preview (filtered)")
        st.dataframe(filtered.head(200))

        # download filtered csv
        st.download_button("Download filtered CSV", data=df_to_csv_bytes(filtered), file_name="filtered_main_data.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Statistical Summary (filtered)")
        try:
            st.dataframe(filtered.describe().T)
        except Exception:
            st.write("No numeric columns to show summary.")

        st.markdown("---")
        st.subheader("Korelasi Fitur (filtered)")
        try:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.heatmap(filtered.corr(numeric_only=True), cmap='vlag', center=0, ax=ax)
            st.pyplot(fig)
        except Exception:
            st.write("Could not compute correlation matrix.")

        st.markdown("---")
        st.subheader("Visualisasi Interaktif (filtered)")
        # monthly
        if cnt and 'mnth' in filtered.columns:
            month_avg = filtered.groupby('mnth')[cnt].mean().reset_index().sort_values('mnth')
            fig = px.bar(month_avg, x='mnth', y=cnt, labels={'mnth':'Month', cnt:'Avg rentals'}, title="Avg rentals by month (filtered)", hover_data={cnt:':.0f'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            if 'mnth' not in filtered.columns:
                st.info("Monthly visualization skipped: 'mnth' missing.")
            else:
                st.info("Monthly visualization skipped: rental count column not detected.")

        # hourly
        if cnt and 'hr' in filtered.columns:
            hour_avg = filtered.groupby('hr')[cnt].mean().reset_index()
            fig = px.line(hour_avg, x='hr', y=cnt, markers=True, title="Avg rentals by hour (filtered)", labels={'hr':'Hour', cnt:'Avg rentals'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            if 'hr' not in filtered.columns:
                st.info("Hourly visualization skipped: 'hr' missing.")
            else:
                st.info("Hourly visualization skipped: rental count column not detected.")

        # weather
        if cnt and 'weathersit' in filtered.columns:
            fig = px.box(filtered, x='weathersit', y=cnt, points="outliers", title="Weather impact on rentals (filtered)")
            st.plotly_chart(fig, use_container_width=True)

        # temp vs rentals
        if cnt and 'temp' in filtered.columns:
            sample = filtered.sample(min(len(filtered), 2000), random_state=42) if len(filtered) > 2000 else filtered
            fig = px.scatter(sample, x='temp', y=cnt, color='mnth' if 'mnth' in sample.columns else None, title="Temp vs rentals (sample)", hover_data=sample.columns)
            st.plotly_chart(fig, use_container_width=True)

        # aggregated tables
        st.markdown("#### Aggregated Tables")
        if cnt in filtered.columns:
            if 'mnth' in filtered.columns:
                st.write("Avg rentals by month")
                st.dataframe(filtered.groupby('mnth')[cnt].mean().reset_index().rename(columns={cnt:'avg_rentals'}))
            if 'hr' in filtered.columns:
                st.write("Avg rentals by hour")
                st.dataframe(filtered.groupby('hr')[cnt].mean().reset_index().rename(columns={cnt:'avg_rentals'}))

    # keep original plot_prediction_analysis functions (unchanged)
    def plot_prediction_analysis(self, prediction, dataset_type):
        tab1, tab2, tab3 = st.tabs(["Prediction Analysis", "Seasonal Trends", "Weather Impact"])
        with tab1:
            self._plot_prediction_gauge(prediction, dataset_type)
        with tab2:
            self._plot_seasonal_trends(dataset_type)
        with tab3:
            self._plot_weather_impact(dataset_type)

    def _plot_prediction_gauge(self, prediction, dataset_type):
        if dataset_type == 'day':
            max_val = 10000
            thresholds = [3000,6000]
        else:
            max_val = 1000
            thresholds = [300,600]
        fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=prediction, title={'text':"Predicted Bike Rentals"},
                                     gauge={'axis':{'range':[None, max_val]}, 'bar':{'color':"#6f42c1"},
                                            'steps':[{'range':[0,thresholds[0]],'color':'#f3e6ff'},{'range':[thresholds[0],thresholds[1]],'color':'#e2cbff'},{'range':[thresholds[1],max_val],'color':'#caa3ff'}],
                                            'threshold':{'line':{'color':'red','width':4},'value':prediction}}))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_seasonal_trends(self, dataset_type):
        seasons = ['Spring','Summer','Fall','Winter']
        avg = [2500,4500,4200,3000] if dataset_type == 'day' else [150,280,260,180]
        fig = px.bar(x=seasons, y=avg, labels={'x':'Season','y':'Avg rentals'}, title="Average Rentals by Season", color=avg, color_continuous_scale='Purples')
        st.plotly_chart(fig, use_container_width=True)

    def _plot_weather_impact(self, dataset_type):
        names = ['Clear','Misty','Light Rain','Heavy Rain']
        vals = [4500,3500,2000,800] if dataset_type=='day' else [280,220,120,50]
        fig = px.pie(values=vals, names=names, title="Weather Impact on Rentals", color_discrete_sequence=px.colors.sequential.Purples)
        st.plotly_chart(fig, use_container_width=True)

    def show_feature_explanations(self):
        with st.expander("Feature Explanations"):
            st.markdown("""
            **Feature Descriptions**
            - Season: 1=Spring, 2=Summer, 3=Fall, 4=Winter  
            - Year: numeric (e.g., 2011, 2012, ...)  
            - Month: 1-12  
            - Hour: 0-23  
            - Holiday: 1=Yes, 0=No  
            - Weekday: 0=Sunday .. 6=Saturday  
            - Working Day: 1=Yes, 0=No  
            - Weather Situation: coded 1..4  
            - Temperature / Feels-like / Humidity / Windspeed: normalized 0-1
            """)

    # main run: use horizontal navbar, preserve original layout and flow
    def run(self):
        # top header + navbar
        colh1, colh2 = st.columns([8,1])
        with colh1:
            st.markdown('<div class="main-title">Bike Sharing Prediction Dashboard</div>', unsafe_allow_html=True)
            st.markdown('<div class="small-muted">Interactive dashboard — summary, visualizations, and Random Forest prediction</div>', unsafe_allow_html=True)
        # horizontal navbar
        selected = self.top_navbar()

        # keep sidebar input (original)
        input_data, dataset_type = self.create_input_sidebar()

        # route by selected page
        if selected == "Ringkasan":
            self.show_data_summary()
            self.show_feature_explanations()
        elif selected == "Visualisasi":
            st.header("Visualisasi Interaktif")
            if self.data is None:
                st.info("Dataset not loaded.")
            else:
                df = self.data
                cnt = self.rental_col
                # overall visuals
                st.subheader("Overview Charts")
                if cnt and 'mnth' in df.columns:
                    month_avg = df.groupby('mnth')[cnt].mean().reset_index().sort_values('mnth')
                    fig = px.bar(month_avg, x='mnth', y=cnt, title="Average rentals by month (overall)", labels={'mnth':'Month', cnt:'Avg rentals'}, color=cnt, color_continuous_scale='Purples')
                    st.plotly_chart(fig, use_container_width=True)
                if cnt and 'hr' in df.columns:
                    hour_avg = df.groupby('hr')[cnt].mean().reset_index()
                    fig = px.line(hour_avg, x='hr', y=cnt, markers=True, title="Average rentals by hour (overall)", labels={'hr':'Hour', cnt:'Avg rentals'})
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Interactive Data Sample")
                st.dataframe(df.sample(min(2000, len(df))).reset_index(drop=True))
                self.show_feature_explanations()
        elif selected == "Prediksi":
            st.header("Analitik & Prediksi")
            st.markdown("Gunakan sidebar untuk mengatur input, lalu klik Predict.")
            if st.button("Predict Bike Rentals", use_container_width=True):
                with st.spinner("Making prediction..."):
                    input_df, display_info = self.preprocess_input(input_data, dataset_type)
                    pred = self.make_prediction(input_df, dataset_type)
                    if pred is None:
                        st.error("Model not loaded or prediction failed.")
                    else:
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        col_a, col_b = st.columns([2,1])
                        with col_a:
                            st.metric("Predicted Bike Rentals", f"{pred:.0f}")
                        with col_b:
                            st.metric("Prediction Type", "Per Day" if dataset_type == 'day' else "Per Hour")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.caption("Predictions for future years are estimated; ensure model trained on representative data for best accuracy.")
                        # show prediction analysis
                        self.plot_prediction_analysis(pred, dataset_type)
                        st.subheader("Input Parameters")
                        st.json(display_info)
        else:
            st.header("About / Info")
            st.markdown("This dashboard loads dataset from GitHub and uses Random Forest models (rf_day_model.pkl & rf_hour_model.pkl) if present.")
            st.markdown("If models are missing, fallback estimates from dataset averages are used.")
            st.markdown("Detected rental count column: **%s**" % (self.rental_col if self.rental_col else "Not detected"))
            if self.data is not None:
                st.write("Dataset shape:", self.data.shape)
            self.show_feature_explanations()

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    dashboard = BikeSharingDashboard()
    dashboard.run()