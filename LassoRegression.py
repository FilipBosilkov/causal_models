import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_df = pd.read_csv("datasets/data_set_4.csv", parse_dates=['Date'])


# Го користиме Price per point како главна променлива, со точка на 300

cutoff = 300

# Create binary treatment based on price cutoff
data_df['Treatment'] = (data_df['Price per Unit'] >= cutoff).astype(int)


left_mask = data_df['Price per Unit'] < cutoff
right_mask = data_df['Price per Unit'] >= cutoff

# Фитуваме на двете страни
left_reg = LinearRegression().fit(
    data_df[left_mask]['Price per Unit'].values.reshape(-1, 1),
    data_df[left_mask]['Total Amount']
)

right_reg = LinearRegression().fit(
    data_df[right_mask]['Price per Unit'].values.reshape(-1, 1),
    data_df[right_mask]['Total Amount']
)

# Пресметуваме ефект кај cutoff
treatment_effect = (right_reg.predict([[cutoff]]) -
                   left_reg.predict([[cutoff]]))

# Ги принтаме резултатите
print(f"Estimated treatment effect at price cutoff ${cutoff}: ${treatment_effect[0]:.2f}")

avg_below = data_df[left_mask]['Total Amount'].mean()
avg_above = data_df[right_mask]['Total Amount'].mean()

print(f"\nAverage sales below ${cutoff}: ${avg_below:.2f}")
print(f"Average sales above ${cutoff}: ${avg_above:.2f}")
print(f"Raw difference: ${avg_above - avg_below:.2f}")

