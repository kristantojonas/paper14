import pandas as pd
from SUB_PREPREPROCESSING import PREPREPROCESSING

# === Load data ===
y_cols = ['Gas', 'Liquid', 'Solid']
X_ops = ['Particle Size (mm)', 'Temperature (C)', 'Residence Time (h)',
         'Carrier Gas (mL/min)', 'Heating Rate (C/h)']
df_raw = pd.read_excel('RAW ML.xlsx', skiprows=1)
df = df_raw[df_raw.columns.difference(['No.', 'First Author'])]
df = PREPREPROCESSING(df)

df_num = df.select_dtypes(include=['number'])

stats_df = pd.DataFrame({
    'Mean': df_num.mean(),
    'Standard Deviation': df_num.std()
})