import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "scikit-learn", "streamlit", "plotly", "joblib"])
