import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import gdown

# ==============================
# 🔧 KONFIGURASI DASBOR
# ==============================
st.set_page_config(
    page_title="Dashboard Analisis Penyewaan Sepeda",
    layout="wide"
)
st.title("🚲 Dashboard Analisis Penyewaan Sepeda")

# ==============================
# 📁 LOAD DATA
# ==============================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/inayahayudeswita/bike-sharing-dashboard/main/dashboard/main_data.csv"
    df = pd.read_csv(url)
    df['dteday'] = pd.to_datetime(df['dteday'])
    day_df = df.drop_duplicates(subset=['dteday'])[['dteday', 'season', 'weathersit_x', 'cnt_x', 'Level_deman_x']]
    hour_df = df[['dteday', 'season', 'hr', 'weathersit_y', 'cnt_y', 'Level_deman_x']]
    return day_df, hour_df

# ==============================
# ⚙️ LOAD MODEL (Dengan fallback dari Google Drive)
# ==============================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_day_path = os.path.join(BASE_DIR, "randomforest_day_model.pkl")
    model_hour_path = os.path.join(BASE_DIR, "randomforest_hour_model.pkl")

    # Link Google Drive (pakai ID file)o
    drive_links = {
        "day": "https://drive.google.com/uc?id=1NICH_ZupZUZM5qCiH5xNXdO6Aab_y0f6",  # day
        "hour": "https://drive.google.com/uc?id=16SUe6L6Pof4gr9O1gBRDpFJq44hSmtTK"  # hour
    }

    # Download kalau belum ada
    if not os.path.exists(model_day_path):
        st.info("Mengunduh model harian dari Google Drive...")
        gdown.download(drive_links["day"], model_day_path, quiet=False)

    if not os.path.exists(model_hour_path):
        st.info("Mengunduh model per jam dari Google Drive...")
        gdown.download(drive_links["hour"], model_hour_path, quiet=False)

    # Load model
    model_day = joblib.load(model_day_path)
    model_hour = joblib.load(model_hour_path)
    return model_day, model_hour


try:
    model_day, model_hour = load_models()
    day_df, hour_df = load_data()
    st.sidebar.success("✅ Data dan model berhasil dimuat!")
    data_loaded = True
except Exception as e:
    st.error(f"Error saat memuat data atau model: {e}")
    st.info("Pastikan file model tersedia atau link Google Drive valid.")
    data_loaded = False

# ==============================
# 🧭 TABS
# ==============================
if data_loaded:
    tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Data", "📈 Visualisasi", "🤖 Prediksi"])

    # =========================
    # 📊 TAB 1 - RINGKASAN DATA
    # =========================
    with tab1:
        st.header("Ringkasan Data Penyewaan Sepeda")
        st.write("### Data Harian")
        st.dataframe(day_df.head())
        st.write("### Data Per Jam")
        st.dataframe(hour_df.head())
        st.write("### Statistik Umum (Data Harian)")
        st.write(day_df.describe())

    # =========================
    # 📈 TAB 2 - VISUALISASI
    # =========================
    with tab2:
        st.sidebar.header("Pengaturan Visualisasi")
        data_view = st.sidebar.radio("Tampilkan Data:", ["Harian (day)", "Per Jam (hour)", "Keduanya"])

        def create_season_pie(df, title, color_palette, count_col):
            summary_df = df.groupby('season')[count_col].sum().reset_index()
            summary_df['season_label'] = summary_df['season'].map({
                1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'
            })
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(summary_df[count_col], labels=summary_df['season_label'],
                   autopct='%1.1f%%', startangle=90, colors=sns.color_palette(color_palette))
            ax.set_title(title)
            ax.axis('equal')
            return fig

        def create_weather_bar(df, title, color_palette, count_col, weather_col):
            weather_df = df.groupby(weather_col)[count_col].sum().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=weather_col, y=count_col, data=weather_df, palette=color_palette, ax=ax)
            ax.set_xlabel("Cuaca")
            ax.set_ylabel("Total Penyewaan")
            ax.set_title(title)
            weather_labels = {1: "Cerah", 2: "Berawan", 3: "Hujan", 4: "Badai"}
            ax.set_xticks(range(len(weather_df)))
            ax.set_xticklabels([weather_labels.get(w, f"Cuaca {w}") for w in weather_df[weather_col]])
            return fig

        if data_view in ["Harian (day)", "Keduanya"]:
            st.subheader("Distribusi Penyewaan Harian")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(create_season_pie(day_df, "Distribusi Penyewaan Berdasarkan Musim (Day)", "pastel", 'cnt_x'))
            with col2:
                st.pyplot(create_weather_bar(day_df, "Penyewaan Berdasarkan Cuaca (Day)", "Blues", 'cnt_x', 'weathersit_x'))

        if data_view in ["Per Jam (hour)", "Keduanya"]:
            st.subheader("Distribusi Penyewaan Per Jam")
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(create_season_pie(hour_df, "Distribusi Berdasarkan Musim (Hour)", "Reds", 'cnt_y'))
            with col2:
                st.pyplot(create_weather_bar(hour_df, "Penyewaan Berdasarkan Cuaca (Hour)", "Greens", 'cnt_y', 'weathersit_y'))

    # =========================
    # 🤖 TAB 3 - PREDIKSI
    # =========================
    with tab3:
        st.header("Prediksi Jumlah Penyewaan Sepeda")
        mode = st.radio("Pilih Mode Prediksi", ["Harian (day)", "Per Jam (hour)"])

        if mode == "Harian (day)":
            st.subheader("Masukkan Parameter Prediksi (Day)")
            col1, col2, col3 = st.columns(3)
            with col1:
                season = st.selectbox("Musim", [1, 2, 3, 4], format_func=lambda x: {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}[x])
                yr = st.selectbox("Tahun (0=2011, 1=2012)", [0, 1])
                mnth = st.slider("Bulan", 1, 12, 6)
            with col2:
                holiday = st.selectbox("Hari Libur", [0, 1])
                weekday = st.slider("Hari ke-", 0, 6, 3)
                workingday = st.selectbox("Hari Kerja", [0, 1])
            with col3:
                weathersit = st.selectbox("Cuaca", [1,2,3,4], format_func=lambda x: {1:'Cerah',2:'Berawan',3:'Hujan',4:'Badai'}[x])
                temp = st.slider("Suhu", 0.0, 1.0, 0.5)
                atemp = st.slider("Suhu Terasa", 0.0, 1.0, 0.5)
                hum = st.slider("Kelembaban", 0.0, 1.0, 0.5)
                windspeed = st.slider("Kecepatan Angin", 0.0, 1.0, 0.2)

            if st.button("Prediksi Jumlah Penyewaan (Day)"):
                input_data = pd.DataFrame([{
                    'season': season, 'yr': yr, 'mnth': mnth, 'holiday': holiday,
                    'weekday': weekday, 'workingday': workingday, 'weathersit': weathersit,
                    'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed
                }])
                pred_day = model_day.predict(input_data)[0]
                st.success(f"🔮 Prediksi Jumlah Penyewaan Harian: **{int(pred_day):,} sepeda**")

        else:
            st.subheader("Masukkan Parameter Prediksi (Hour)")
            col1, col2, col3 = st.columns(3)
            with col1:
                season = st.selectbox("Musim", [1,2,3,4], key="season_hour",
                                      format_func=lambda x: {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}[x])
                yr = st.selectbox("Tahun", [0,1], key="yr_hour")
                mnth = st.slider("Bulan", 1, 12, 6, key="mnth_hour")
            with col2:
                hr = st.slider("Jam", 0, 23, 12, key="hr_hour")
                holiday = st.selectbox("Hari Libur", [0,1], key="hol_hour")
                weekday = st.slider("Hari ke-", 0, 6, 3, key="weekday_hour")
                workingday = st.selectbox("Hari Kerja", [0,1], key="work_hour")
            with col3:
                weathersit = st.selectbox("Cuaca", [1,2,3,4], key="weather_hour",
                                          format_func=lambda x: {1:'Cerah',2:'Berawan',3:'Hujan',4:'Badai'}[x])
                temp = st.slider("Suhu", 0.0, 1.0, 0.5, key="temp_hour")
                atemp = st.slider("Suhu Terasa", 0.0, 1.0, 0.5, key="atemp_hour")
                hum = st.slider("Kelembaban", 0.0, 1.0, 0.5, key="hum_hour")
                windspeed = st.slider("Kecepatan Angin", 0.0, 1.0, 0.2, key="wind_hour")

            if st.button("Prediksi Jumlah Penyewaan (Hour)"):
                input_data = pd.DataFrame([{
                    'season': season, 'yr': yr, 'mnth': mnth, 'hr': hr,
                    'holiday': holiday, 'weekday': weekday, 'workingday': workingday,
                    'weathersit': weathersit, 'temp': temp, 'atemp': atemp,
                    'hum': hum, 'windspeed': windspeed
                }])
                pred_hour = model_hour.predict(input_data)[0]
                st.success(f"🔮 Prediksi Jumlah Penyewaan Per Jam: **{int(pred_hour):,} sepeda**")
                fig, ax = plt.subplots(figsize=(5,3))
                ax.bar(["Prediksi"], [pred_hour], color="skyblue")
                ax.set_ylabel("Jumlah Penyewaan")
                ax.set_title("Visualisasi Hasil Prediksi")
                st.pyplot(fig)
