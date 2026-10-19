# -*- coding: utf-8 -*-
"""
Generate synthetic common prosperity time-series data
31 provinces x 30 years x 8 indicators
Output: data/common_prosperity_data.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)

n_provinces = 31
n_years = 30
start_year = 1995

provinces = [
    'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'InnerMongolia', 'Liaoning', 'Jilin', 'Heilongjiang',
    'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian', 'Jiangxi', 'Shandong', 'Henan',
    'Hubei', 'Hunan', 'Guangdong', 'Guangxi', 'Hainan', 'Chongqing', 'Sichuan', 'Guizhou',
    'Yunnan', 'Tibet', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
]

indicators = [
    'GDP_per_capita',          # positive
    'Disposable_income',       # positive
    'Urbanization_rate',       # positive
    'Education_ratio',         # positive
    'Social_security',         # positive
    'Urban_rural_income_ratio',# negative (lower = better)
    'Environment_index',       # positive
    'Infrastructure_index',    # positive
]

rows = []

for p_idx, prov in enumerate(provinces):
    base_level = 0.2 + 0.5 * (1 - p_idx / n_provinces) + np.random.uniform(-0.05, 0.05)
    growth_rate = 0.02 + 0.015 * np.random.random()

    for y_idx, year in enumerate(range(start_year, start_year + n_years)):
        t = y_idx / (n_years - 1)

        trend = base_level + (1 - base_level) * (1 - np.exp(-growth_rate * y_idx / 5))
        cycle = 0.03 * np.sin(2 * np.pi * y_idx / 7 + p_idx)
        noise = np.random.normal(0, 0.015, 8)

        values = []

        # 1. GDP per capita
        v1 = (trend + 0.1 * t + cycle + noise[0])
        values.append(np.clip(v1, 0.1, 0.98))

        # 2. Disposable income
        v2 = 0.8 * v1 + 0.05 * t + cycle * 0.5 + noise[1] * 0.8
        values.append(np.clip(v2, 0.1, 0.98))

        # 3. Urbanization rate - S-curve
        v3 = 0.3 + 0.5 / (1 + np.exp(-6 * (t - 0.4))) + noise[2] * 0.5
        values.append(np.clip(v3, 0.15, 0.95))

        # 4. Education ratio
        v4 = 0.2 + 0.4 * t ** 0.8 + noise[3] * 0.6
        values.append(np.clip(v4, 0.1, 0.9))

        # 5. Social security coverage
        v5 = 0.1 + 0.7 * t ** 1.5 + noise[4] * 0.4
        values.append(np.clip(v5, 0.1, 0.98))

        # 6. Urban-rural income ratio - negative
        v6 = 2.0 + 1.0 * np.sin(np.pi * t)
        v6 = (v6 - 1.5) / 2.0
        values.append(np.clip(v6, 0.05, 0.95))

        # 7. Environment index - EKC pattern
        v7 = 0.7 - 0.3 * np.sin(np.pi * t) + noise[6] * 0.4
        values.append(np.clip(v7, 0.1, 0.95))

        # 8. Infrastructure index
        v8 = 0.15 + 0.65 * t ** 1.2 + noise[7] * 0.5
        values.append(np.clip(v8, 0.1, 0.95))

        rows.append({
            'Province': prov,
            'Year': year,
            **dict(zip(indicators, values))
        })

df = pd.DataFrame(rows)

# Calculate Common Prosperity Index (simple equal weight, invert negative)
positive = ['GDP_per_capita', 'Disposable_income', 'Urbanization_rate',
            'Education_ratio', 'Social_security', 'Environment_index',
            'Infrastructure_index']
negative = ['Urban_rural_income_ratio']

for col in positive:
    df[col + '_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
for col in negative:
    df[col + '_norm'] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())

norm_cols = [c + '_norm' for c in indicators]
df['CPI'] = df[norm_cols].mean(axis=1)

# Save
output_path = 'data/common_prosperity_data.csv'
df.to_csv(output_path, index=False)

print(f"Data generated successfully!")
print(f"  Samples: {len(df)} ({n_provinces} provinces x {n_years} years)")
print(f"  Indicators: {len(indicators)}")
print(f"  Saved to: {output_path}")
print(f"  CPI range: [{df['CPI'].min():.4f}, {df['CPI'].max():.4f}]")
print(f"  CPI mean:  {df['CPI'].mean():.4f}")
