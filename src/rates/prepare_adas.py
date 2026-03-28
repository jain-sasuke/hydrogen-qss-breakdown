import pandas as pd

# SCD — rename K_cm3_s → SCD_cm3_s
scd = pd.read_csv('data/processed/adas/scd96_h_long.csv')
scd = scd.rename(columns={'K_cm3_s': 'SCD_cm3_s'})
scd.to_csv('data/processed/adas/SCD96_interpolated.csv', index=False)
print('SCD columns:', scd.columns.tolist())
print('SCD shape:', scd.shape)

# ACD — rename K_cm3_s → ACD_cm3_s
acd = pd.read_csv('data/processed/adas/acd96_h_long.csv')
acd = acd.rename(columns={'K_cm3_s': 'ACD_cm3_s'})
acd.to_csv('data/processed/adas/ACD96_interpolated.csv', index=False)
print('ACD columns:', acd.columns.tolist())
print('ACD shape:', acd.shape)

print('Done — files written to data/processed/adas/')