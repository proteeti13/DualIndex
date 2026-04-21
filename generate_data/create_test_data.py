import random
import csv

def generate_graph_data(num_edges=100, node_range=500):
    # 1. Generate Data: Unique pairs of (Source, Dest)
    # Using a set to ensure all edges are unique
    edges_set = set()
    while len(edges_set) < num_edges:
        src = float(random.randint(0, node_range))
        dest = float(random.randint(0, node_range))
        # Prevent self-loops (optional, but standard for most graph tests)
        if src != dest:
            edges_set.add((src, dest))
    
    # 2. Crucial Sorting: Primary (Source), Secondary (Dest)
    # This creates the monotonic order required for learned index models
    sorted_edges = sorted(list(edges_set), key=lambda x: (x[0], x[1]))
    
    # 3. Assign Offsets: Sequential integers 0...N
    # 4. Output Format: Save as graph_data.csv (raw data, no header)
    output_file = "graph_data.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i, (src, dest) in enumerate(sorted_edges):
            # Target format: SourceID, DestID, GlobalOffset
            # Offset is typed as a 64-bit integer conceptually
            writer.writerow([src, dest, i])

    # 5. Verification: Print first 10 rows
    print(f"Successfully generated {num_edges} edges in '{output_file}'.")
    print("-" * 45)
    print(f"{'Source_ID':<12} | {'Dest_ID':<12} | {'Global_Offset'}")
    print("-" * 45)
    for i in range(10):
        src, dest = sorted_edges[i]
        print(f"{src:<12.1f} | {dest:<12.1f} | {i}")

if __name__ == "__main__":
    # You can adjust the number of edges here
    generate_graph_data(num_edges=75)