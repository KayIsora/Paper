import pandas as pd

# Read both CSV files
auc_compare = pd.read_csv('auc_compare.csv')
auc_prior2_base = pd.read_csv('auc_prior2_base.csv')

# Get all 'update' columns from auc_compare
update_columns = [col for col in auc_compare.columns if 'update' in col.lower()]
print(f"Update columns found: {update_columns}")

# Create a new dataframe starting with auc_prior2_base
result = auc_prior2_base.copy()

# Replace all 'update' columns with data from auc_compare
for col in update_columns:
    if col in result.columns:
        result[col] = auc_compare[col]
        print(f"Replaced column: {col}")

# Recalculate delta_auc if it exists
if 'delta_auc' in result.columns and 'auc_update' in result.columns and 'auc_pure' in result.columns:
    result['delta_auc'] = result['auc_update'] - result['auc_pure']
    print("Recalculated delta_auc")

# Save to a new file
result.to_csv('auc_prior2_base.csv', index=False)
print("\nSuccessfully merged! auc_prior2_base.csv has been updated.")
print(f"\nFirst 5 rows of updated file:")
print(result.head())
