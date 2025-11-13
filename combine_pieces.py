import pickle
import glob
import os

# Find all pkl files you want to combine
pkl_files = glob.glob('piece_values/piece_values*.pkl')  # adjust the pattern if needed

combined_data = {}

for file in pkl_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    for pt, vals in data.items():
        combined_data.setdefault(pt, []).extend(vals)

# Save the merged data
with open('combined_piece_values.pkl', 'wb') as f:
    pickle.dump(combined_data, f)

print(f'Combined {len(pkl_files)} files into combined_piece_values.pkl')