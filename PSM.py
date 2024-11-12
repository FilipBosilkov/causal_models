import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


scanner_data_path = "datasets/data_set_2.csv"
scanner_data_df = pd.read_csv(scanner_data_path, index_col=0)


# X52 и OH2 од SKU категоријата, се дел од групата која ја тестираме.
scanner_data_df['Treatment_Group'] = scanner_data_df['SKU_Category'].isin(['X52', '0H2']).astype(int)

# Додади бинарни колони
matching_df = pd.get_dummies(scanner_data_df[['Customer_ID', 'Quantity', 'SKU_Category', 'Treatment_Group']],
                             columns=['SKU_Category'], drop_first=True)

# Х ни е контролната група додека пак Y ни е главната група
X = matching_df.drop(columns='Treatment_Group')
y = matching_df['Treatment_Group']

# го фитуваме моделот за да пресметаме propensity score
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, y)
matching_df['Propensity_Score'] = log_reg.predict_proba(X)[:, 1]

# Ги делиме на групи што се тестирани и оние што не се
treated = matching_df[matching_df['Treatment_Group'] == 1]
control = matching_df[matching_df['Treatment_Group'] == 0]

# со NearestNeighbors ги поврзуваме оние што се најслични
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control[['Propensity_Score']])
distances, indices = nn.kneighbors(treated[['Propensity_Score']])

# пониско ниво на листа
matched_control_indices = indices.flatten()
# се избираат редовите што се совпаѓаат
matched_control = control.iloc[matched_control_indices].copy()
# се додавава Matched колона на двете
treated.loc[:, 'Matched'] = True
matched_control.loc[:, 'Matched'] = True

matched_df = pd.concat([treated, matched_control])

# ги спојуваме податоците со оригиналните за да може да ги споредиме
merged_data = scanner_data_df.merge(matched_df[['Customer_ID', 'Quantity', 'Treatment_Group', 'Matched']],
                                    on=['Customer_ID', 'Quantity', 'Treatment_Group'], how='inner')

treated_sales = merged_data[merged_data['Treatment_Group'] == 1]['Sales_Amount'].mean()
control_sales = merged_data[merged_data['Treatment_Group'] == 0]['Sales_Amount'].mean()
treatment_effect = treated_sales - control_sales


print(merged_data)

print(treated_sales, control_sales, treatment_effect)
