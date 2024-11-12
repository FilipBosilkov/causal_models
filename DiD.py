import pandas as pd
import statsmodels.api as sm

warehouse_retail_sales_path = "datasets/data_set_1.csv"
warehouse_retail_sales_df = pd.read_csv(warehouse_retail_sales_path)

intervention_start_year = 2019
intervention_start_month = 2

# Ги зачувуваме сите податоци што се после интеревенцијата
warehouse_retail_sales_df['Post_Intervention'] = (
    (warehouse_retail_sales_df['YEAR'] > intervention_start_year) |
    ((warehouse_retail_sales_df['YEAR'] == intervention_start_year) &
     (warehouse_retail_sales_df['MONTH'] >= intervention_start_month))
).astype(int)

# Главната група се сите редови со тип Вино
warehouse_retail_sales_df['Treatment_Group'] = (warehouse_retail_sales_df['ITEM TYPE'] == 'WINE').astype(int)

# Ги собираме соодветните суми
did_summary = warehouse_retail_sales_df.groupby(['YEAR', 'MONTH', 'Treatment_Group', 'Post_Intervention']).agg(
    Total_Sales=('RETAIL SALES', 'sum')
).reset_index()


print(did_summary)


# Прво ја дефинираме интеракцијата - Treatment_Group и Post_Intervention
# Потоа извршуваме регресија
did_summary['DiD_Interaction'] = did_summary['Treatment_Group'] * did_summary['Post_Intervention']

X = did_summary[['Treatment_Group', 'Post_Intervention', 'DiD_Interaction']]
X = sm.add_constant(X)
y = did_summary['Total_Sales']


model = sm.OLS(y, X).fit()
did_results = model.summary()


print(did_results)

