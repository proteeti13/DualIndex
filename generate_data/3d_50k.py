import numpy as np
import pandas as pd

def generate_3d_path_data(num_records=50000, file_name="3d_50k_graph_paths.csv"):
    # 1. Generate random triplets of floating-point numbers
    # We use a range (e.g., 0 to 1000) to simulate Node IDs
    print(f"Generating {num_records} random 3D paths...")
    data = np.random.uniform(0, 1000, size=(num_records, 3))
    
    # 2. Convert to a DataFrame for easier sorting logic
    df = pd.DataFrame(data, columns=['Source_ID', 'Hop1_ID', 'Hop2_ID'])
    
    # 3. CRITICAL: Hierarchical Lexicographical Sort
    # Priority: Source_ID -> Hop1_ID -> Hop2_ID
    print("Sorting data lexicographically to build a learnable CDF...")
    df = df.sort_values(by=['Source_ID', 'Hop1_ID', 'Hop2_ID']).reset_index(drop=True)
    
    # 4. Add the Global Offset (Label)
    # In a learned index, the model predicts this index/position
    df['Global_Offset'] = df.index
    
    # 5. Save the data
    # .csv is good for inspection, but 'learnedbench' often prefers binary (.bin)
    df.to_csv(file_name, index=False)
    print(f"Success! Data saved to {file_name}")
    
    # Quick verification of the first few rows
    print("\nSample of Sorted Data:")
    print(df.head())

if __name__ == "__main__":
    generate_3d_path_data()