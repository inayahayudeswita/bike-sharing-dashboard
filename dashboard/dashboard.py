import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import plotly.express as px
import os

st.set_page_config(
    page_title="Dashboard Analisis Penyewaan Sepeda",
    layout="wide",
)
st.title("🚲 Dashboard Analisis Penyewaan Sepeda")

SEASON_MAP = {1: 'Musim Semi', 2: 'Musim Panas', 3: 'Musim Gugur', 4: 'Musim Dingin'}
WEATHER_LABELS = {
    1: 'Cerah',
    2: 'Mendung',
    3: 'Hujan / Salju',
    4: 'Hujan Deras/ Salju Lebat'
}

DAYNAME_MAP = {
    'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
    'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
}
WEEKDAY_ORDER = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']

PALETTE = ["#1E7ACB", "#3795DE", "#58A9E6", "#84C4F0", "#B5E1FA"]

# Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/inayahayudeswita/bike-sharing-dashboard/refs/heads/main/data/main_data.csv"
    df = pd.read_csv(url)
    df['dteday'] = pd.to_datetime(df['dteday'])

    day_df = df.drop_duplicates(subset=['dteday'])[['dteday', 'season', 'weathersit_x', 'cnt_x', 'Level_deman_x']]
    hour_df = df[['dteday', 'season', 'hr', 'weathersit_y', 'cnt_y', 'Level_deman_x']]

    day_df = day_df.rename(columns={'weathersit_x':'weathersit','cnt_x':'cnt','Level_deman_x':'level'})
    hour_df = hour_df.rename(columns={'weathersit_y':'weathersit','cnt_y':'cnt','Level_deman_x':'level'})

    day_df['weekday_name'] = day_df['dteday'].dt.day_name().map(DAYNAME_MAP)
    hour_df['weekday_name'] = hour_df['dteday'].dt.day_name().map(DAYNAME_MAP)
    return day_df, hour_df

# Load Scaler dan Model
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_paths = {
        "scaler_day": os.path.join(BASE_DIR, "day_scaler.pkl"),
        "scaler_hour": os.path.join(BASE_DIR, "hour_scaler.pkl"),
        "model_day": os.path.join(BASE_DIR, "rf_day_model.pkl"),
        "model_hour": os.path.join(BASE_DIR, "rf_hour_model.pkl")
    }
    loaded_files = {}
    for key, path in file_paths.items():
        if not os.path.exists(path):
            file_name = os.path.basename(path)
            raise FileNotFoundError(
                f"File **{file_name}** tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan dashboard.py."
            )
        loaded_files[key] = joblib.load(path)

    return (
        loaded_files["scaler_day"],
        loaded_files["scaler_hour"],
        loaded_files["model_day"],
        loaded_files["model_hour"]
    )

# HELPER : Selaraskan fitur ke caler/model
def align_features_for(estimator, df_input):
    df = df_input.copy()

    expected = getattr(estimator, "feature_names_in_", None)
    if expected is None:
        return df

    expected = list(expected)

    default_numeric = {
        "yr": 1,              
        "holiday": 0,
        "workingday": 1,

        "season": 1,
        "mnth": 6,
        "weekday": 3,
        "weathersit": 1,
        "hr": 12,

        "temp": 0.5,
        "atemp": 0.5,
        "hum": 0.5,
        "windspeed": 0.2,
    }

    for col in expected:
        if col not in df.columns:
            df[col] = default_numeric.get(col, 0.0)

    df = df[expected]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default_numeric.get(c, 0.0))

    return df

def debug_expected_vs_provided(estimator, df_input, where=""):
    expected = list(getattr(estimator, "feature_names_in_", []))
    missing = [c for c in expected if c not in df_input.columns]
    extra = [c for c in df_input.columns if c not in expected]
    if missing or extra:
        st.warning(f"[DEBUG {where}] Fitur hilang: {missing} | Fitur ekstra: {extra}")

# Init Data dan Model
try:
    scaler_day, scaler_hour, model_day, model_hour = load_models()
    day_df, hour_df = load_data()
    st.sidebar.success("✅ Data, scaler, dan model berhasil dimuat.")
    data_loaded = True
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data/scaler/model: {e}")
    st.info("Pastikan file: day_scaler.pkl, hour_scaler.pkl, rf_day_model.pkl, rf_hour_model.pkl berada di folder yang sama dengan dashboard.py.")
    data_loaded = False

# Sidebar (Fitur Filter)
if data_loaded:
    with st.sidebar:
        st.header("Filter")
        min_date, max_date = day_df["dteday"].min().date(), day_df["dteday"].max().date()
        start_date, end_date = st.date_input(
            "Rentang Tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        seasons = st.multiselect(
            "Musim",
            options=sorted(SEASON_MAP.keys()),
            default=sorted(SEASON_MAP.keys()),
            format_func=lambda x: SEASON_MAP[x],
        )
        weathers = st.multiselect(
            "Cuaca",
            options=sorted(day_df["weathersit"].unique()),
            default=sorted(day_df["weathersit"].unique()),
            format_func=lambda x: WEATHER_LABELS.get(x, str(x)),
        )

    mask_day = (
        (day_df["dteday"].dt.date >= start_date)
        & (day_df["dteday"].dt.date <= end_date)
        & (day_df["season"].isin(seasons))
        & (day_df["weathersit"].isin(weathers))
    )
    mask_hour = (
        (hour_df["dteday"].dt.date >= start_date)
        & (hour_df["dteday"].dt.date <= end_date)
        & (hour_df["season"].isin(seasons))
        & (hour_df["weathersit"].isin(weathers))
    )
    md = day_df.loc[mask_day].copy()
    mh = hour_df.loc[mask_hour].copy()

# Tab Dashboard
if data_loaded:
    tab_summary, tab_viz, tab_pred = st.tabs(["Overeview Data", "Visualisasi Data", "Prediksi"])

    # Tab 1 - Ringkasan Data
    with tab_summary:
        st.subheader("Ringkasan Data")
        st.caption("Angka di bawah dihitung dari data yang sudah difilter (tanggal, musim, dan cuaca) pada sidebar.")

        # Tertinggi & Terendah (Harian) 
        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Penyewaan Harian")
            if md.empty:
                st.info("Tidak ada data harian pada rentang filter.")
            else:
                idx_max_day = md['cnt'].idxmax()
                idx_min_day = md['cnt'].idxmin()
                row_max_day = md.loc[idx_max_day]
                row_min_day = md.loc[idx_min_day]

                m1, m2 = st.columns(2)
                m1.metric("Tertinggi (Harian)", f"{int(row_max_day['cnt']):,}", f"{row_max_day['dteday'].date()}")
                m2.metric("Terendah (Harian)", f"{int(row_min_day['cnt']):,}", f"{row_min_day['dteday'].date()}")

        # Tertinggi & Terendah (Per Jam) 
        with colB:
            st.markdown("#### Penyewaan per Jam")
            if mh.empty:
                st.info("Tidak ada data per jam pada rentang filter.")
            else:
                mh_disp = mh.copy()
                mh_disp['timestamp'] = mh_disp['dteday'] + pd.to_timedelta(mh_disp['hr'], unit='h')

                idx_max_hr = mh_disp['cnt'].idxmax()
                idx_min_hr = mh_disp['cnt'].idxmin()
                row_max_hr = mh_disp.loc[idx_max_hr]
                row_min_hr = mh_disp.loc[idx_min_hr]

                m3, m4 = st.columns(2)
                m3.metric("Tertinggi (Jam)", f"{int(row_max_hr['cnt']):,}", row_max_hr['timestamp'].strftime("%Y-%m-%d %H:00"))
                m4.metric("Terendah (Jam)", f"{int(row_min_hr['cnt']):,}", row_min_hr['timestamp'].strftime("%Y-%m-%d %H:00"))

        st.markdown("---")

        # Rata-rata Hari / Minggu / Tahun 
        st.markdown("#### Rata-rata Penyewaan")
        if md.empty:
            st.info("Tidak ada data harian untuk menghitung rata-rata.")
        else:
            avg_per_day = float(md['cnt'].mean())

            weekly_total = md.set_index('dteday')['cnt'].resample('W').sum()
            avg_per_week = float(weekly_total.mean()) if not weekly_total.empty else 0.0

            yearly_total = md.set_index('dteday')['cnt'].resample('Y').sum()
            avg_per_year = float(yearly_total.mean()) if not yearly_total.empty else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Rata-rata per Hari", f"{avg_per_day:,.0f}")
            c2.metric("Rata-rata per Minggu", f"{avg_per_week:,.0f}")
            c3.metric("Rata-rata per Tahun", f"{avg_per_year:,.0f}")

            st.caption("Metode: Hari = rata-rata langsung; Minggu/Tahun = total per periode (resample) lalu dirata-ratakan antarminggu/tahun.")

        st.markdown("---")

        # Preview Dataset 
        st.markdown("#### Pratinjau Dataset (setelah filter)")
        cL, cR = st.columns(2)
        with cL:
            st.markdown(f"**Data Harian (md)** — {len(md):,} baris")
            st.dataframe(md.head(15), use_container_width=True)
        with cR:
            st.markdown(f"**Data per Jam (mh)** — {len(mh):,} baris")
            st.dataframe(mh.head(15), use_container_width=True)

    # Tab 2 - Visualisasi Data
    with tab_viz:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Rata-rata Penyewaan per Hari")
            weekday_avg = md.groupby("weekday_name", as_index=False)["cnt"].mean()
            weekday_avg["weekday_name"] = pd.Categorical(weekday_avg["weekday_name"], categories=WEEKDAY_ORDER, ordered=True)
            weekday_avg = weekday_avg.sort_values("weekday_name", ascending=True)

            fig_wd = px.bar(
                weekday_avg,
                y="weekday_name",
                x="cnt",
                orientation="h",
                labels={"weekday_name": "Hari", "cnt": "Rata-rata Penyewaan"},
                template="plotly_white",
                title="Rata-rata Penyewaan per Hari ",
                color_discrete_sequence=PALETTE,
            )
            fig_wd.update_layout(
                yaxis=dict(categoryorder="array", categoryarray=WEEKDAY_ORDER),
                margin=dict(l=80, r=40, t=60, b=40),
            )
            st.plotly_chart(fig_wd, use_container_width=True, theme=None)

        with col2:
            st.markdown("### Tren Penyewaan per Jam")
            hourly_avg = mh.groupby("hr", as_index=False)["cnt"].mean().sort_values("hr")
            fig_hr = px.bar(
                hourly_avg,
                x="hr",
                y="cnt",
                template="plotly_white",
                title="Tren Penyewaan per Jam",
                color_discrete_sequence=PALETTE,
            )
            fig_hr.update_xaxes(dtick=1, tick0=0, title_text="Jam")
            fig_hr.update_yaxes(title_text="Rata-rata Penyewaan")
            st.plotly_chart(fig_hr, use_container_width=True, theme=None)

        st.markdown("### Tren Penyewaan Bulanan")
        monthly = md.set_index("dteday")["cnt"].resample("M").sum().reset_index()
        monthly["periode"] = monthly["dteday"].dt.to_period("M").astype(str)
        fig_month = px.line(
            monthly,
            x="periode",
            y="cnt",
            markers=True,
            title="Tren Penyewaan Bulanan",
            template="plotly_white",
            color_discrete_sequence=PALETTE,
            labels={"periode": "Periode (YYYY-MM)", "cnt": "Total Penyewaan"}
        )

        fig_month.update_layout(title=dict(x=0.0, xanchor="left"))
        fig_month.update_xaxes(tickangle=45)
        st.plotly_chart(fig_month, use_container_width=True, theme=None)

        st.markdown("### Heatmap — Rata-rata Penyewaan (Hari × Jam)")
        mh_disp2 = mh.copy()
        mh_disp2["Hari"] = mh_disp2["weekday_name"]
        heat = mh_disp2.pivot_table(index="Hari", columns="hr", values="cnt", aggfunc="mean")
        heat = heat.reindex(WEEKDAY_ORDER) 
        fig_heat = px.imshow(
            heat,
            aspect="auto",
            origin="lower",
            labels=dict(x="Jam", y="Hari", color="Rata-rata"),
            title="Heatmap Rata-rata Penyewaan (Jam × Hari)",
            color_continuous_scale=PALETTE,
        )
        fig_heat.update_layout(title=dict(x=0.0, xanchor="left"))
        st.plotly_chart(fig_heat, use_container_width=True, theme=None)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Penyewaan menurut Cuaca (Harian)")
            wday = (
                md.groupby("weathersit", as_index=False)["cnt"].sum()
                  .assign(Cuaca=lambda d: d["weathersit"].map(WEATHER_LABELS))
            )
            fig_wday = px.bar(
                wday, x="Cuaca", y="cnt",
                template="plotly_white",
                title="Penyewaan menurut Cuaca (Harian)",
                color_discrete_sequence=PALETTE,
                labels={"cnt": "Total Penyewaan"}
            )
            fig_wday.update_layout(title=dict(x=0.0, xanchor="left"))
            st.plotly_chart(fig_wday, use_container_width=True, theme=None)
        with col4:
            st.markdown("### Penyewaan menurut Musim (Harian)")
            sday = (
                md.groupby("season", as_index=False)["cnt"].sum()
                  .assign(Musim=lambda d: d["season"].map(SEASON_MAP))
            )
            fig_sday = px.pie(
                sday,
                names="Musim",
                values="cnt",
                hole=0.35,
                title="Distribusi Penyewaan menurut Musim (Harian)",
                color_discrete_sequence=PALETTE,
            )
            fig_sday.update_layout(title=dict(x=0.0, xanchor="left"))
            st.plotly_chart(fig_sday, use_container_width=True, theme=None)

        st.markdown("### Korelasi Antar Kolom (Pearson) — Data Harian")
        st.caption("Korelasi menunjukkan seberapa kuat hubungan linear antar variabel numerik pada data harian.")

        if not md.empty:
            numeric_cols = md.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) >= 2:
                corr = md[numeric_cols].corr(method="pearson")
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=PALETTE,
                    labels=dict(color="Korelasi"),
                )
                fig_corr.update_layout(
                    title=dict(
                        text="Heatmap Korelasi Fitur (Metode Pearson)",
                        x=0.0, xanchor="left",
                        font=dict(size=18, color="#333", family="Arial"),
                    ),
                    margin=dict(l=60, r=30, t=70, b=30),
                )
                st.plotly_chart(fig_corr, use_container_width=True, theme=None)

                # Insight otomatis (3 terkuat)
                corr_abs = corr.abs().copy()
                pairs = []
                cols = corr_abs.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i+1, len(cols)):
                        pairs.append((cols[i], cols[j], corr.loc[cols[i], cols[j]], corr_abs.loc[cols[i], cols[j]]))
                if pairs:
                    top3 = sorted(pairs, key=lambda x: x[3], reverse=True)[:3]
                    st.markdown("#### 💡 Korelasi Terkuat yang Menarik")
                    for a, b, val, _ in top3:
                        arah = "positif" if val >= 0 else "negatif"
                        st.write(f"- **{a}** ↔ **{b}**: korelasi {arah} **{val:.2f}**")
                    st.caption("Mendekati 1/-1 = sangat kuat; mendekati 0 = lemah.")
            else:
                st.info("Kolom numerik tidak cukup untuk menghitung korelasi.")
        else:
            st.info("Tidak ada data untuk menampilkan korelasi.")

        st.caption("Semua visual mengikuti filter. Bulanan memakai resample('M'); hari urut Senin→Minggu; palet satu-tone biru.")

    # Tab 3 - Prediksi
    with tab_pred:
        st.header("Prediksi Jumlah Penyewaan Sepeda")
        mode = st.radio("Pilih Mode Prediksi", ["Harian (day)", "Per Jam (hour)"])

        default_env = {"temp": 0.5, "atemp": 0.5, "hum": 0.5, "windspeed": 0.2}

        if mode == "Harian (day)":
            st.subheader("Masukkan Parameter Prediksi (Harian)")
            col1, col2 = st.columns(2)
            with col1:
                season = st.selectbox("Musim", [1, 2, 3, 4], format_func=lambda x: SEASON_MAP[x])
                mnth = st.slider("Bulan", 1, 12, 6)
            with col2:
                holiday = st.selectbox("Hari Libur", [0, 1])
                weekday = st.slider("Hari ke- (0=Senin ... 6=Minggu)", 0, 6, 3)
                workingday = st.selectbox("Hari Kerja", [0, 1])
                weathersit = st.selectbox("Kondisi Cuaca", [1, 2, 3, 4],
                                          format_func=lambda x: {1:'Cerah',2:'Berawan',3:'Hujan',4:'Badai'}[x])
            
            day_features_ui = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

            if st.button("Prediksi Jumlah Penyewaan (Harian)"):
                input_data = pd.DataFrame([{
                    'season': season, 'mnth': mnth, 'holiday': holiday,
                    'weekday': weekday, 'workingday': workingday, 'weathersit': weathersit,
                    **default_env  
                }])

                debug_expected_vs_provided(scaler_day, input_data, where="DAY-SCALER (sebelum align)")

                input_for_scaler = align_features_for(scaler_day, input_data)
                input_scaled = scaler_day.transform(input_for_scaler)
                input_scaled_df = pd.DataFrame(input_scaled, columns=getattr(scaler_day, "feature_names_in_", input_for_scaler.columns))
                input_for_model = align_features_for(model_day, input_scaled_df)

                pred_day = model_day.predict(input_for_model)[0]
                st.success(f"🔮 Prediksi jumlah penyewaan harian: **{int(pred_day):,} sepeda**")

        else:
            st.subheader("Masukkan Parameter Prediksi (Per Jam)")
            col1, col2 = st.columns(2)
            with col1:
                season = st.selectbox("Musim", [1, 2, 3, 4], key="season_hour", format_func=lambda x: SEASON_MAP[x])
                mnth = st.slider("Bulan", 1, 12, 6, key="mnth_hour")
                hr = st.slider("Jam", 0, 23, 12, key="hr_hour")
            with col2:
                holiday = st.selectbox("Hari Libur", [0, 1], key="hol_hour")
                weekday = st.slider("Hari ke- (0=Senin ... 6=Minggu)", 0, 6, 3, key="weekday_hour")
                workingday = st.selectbox("Hari Kerja", [0, 1], key="work_hour")
                weathersit = st.selectbox("Kondisi Cuaca", [1, 2, 3, 4], key="weather_hour",
                                          format_func=lambda x: {1:'Cerah',2:'Berawan',3:'Hujan',4:'Badai'}[x])

            hour_features_ui = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']

            if st.button("Prediksi Jumlah Penyewaan (Per Jam)"):
                input_data = pd.DataFrame([{
                    'season': season, 'mnth': mnth, 'hr': hr,
                    'holiday': holiday, 'weekday': weekday, 'workingday': workingday,
                    'weathersit': weathersit,
                    **default_env  
                }])

                debug_expected_vs_provided(scaler_hour, input_data, where="HOUR-SCALER (sebelum align)")

                input_for_scaler = align_features_for(scaler_hour, input_data)
                input_scaled = scaler_hour.transform(input_for_scaler)
                input_scaled_df = pd.DataFrame(input_scaled, columns=getattr(scaler_hour, "feature_names_in_", input_for_scaler.columns))
                input_for_model = align_features_for(model_hour, input_scaled_df)

                pred_hour = model_hour.predict(input_for_model)[0]
                st.success(f"🔮 Prediksi jumlah penyewaan per jam: **{int(pred_hour):,} sepeda**")

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(["Prediksi"], [pred_hour], color=PALETTE[2])
                ax.set_ylabel("Jumlah Penyewaan")
                ax.set_title("Visualisasi Hasil Prediksi")
                st.pyplot(fig)
