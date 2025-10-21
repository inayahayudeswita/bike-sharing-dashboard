# Dashboard Analisis Penyewaan Sepeda

Dashboard ini dirancang untuk menganalisis data penyewaan sepeda berdasarkan berbagai faktor seperti musim, cuaca, dan waktu. Aplikasi ini dibangun menggunakan Streamlit dan Python.

## Setup Environment

### Menggunakan Anaconda

1. Setup Environment - Anaconda
   conda create --name main-ds python=3.13.2
   conda activate main-ds
   pip install -r requirements.txt

2. Setup Environment - Shell/Terminalz
    mkdir dashboard_penyewaan_sepeda
    cd dashboard_penyewaan_sepeda
    pipenv install
    pipenv shell
    pip install -r requirements.txt

3. Run steamlit app
    cd dashboard
    streamlit run dashboard.py