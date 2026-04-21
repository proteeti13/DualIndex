import struct
import os

# CONFIGURATION
INPUT_FILE = "graph_data.bin"
OUTPUT_FILE = "graph_data_tpie.bin"
NUM_POINTS = 75
DIM = 2

# CRITICAL: TPIE stores INDIVIDUAL DOUBLES, not complete points!
ITEM_SIZE = 8  # Size of a single double
NUM_ITEMS = NUM_POINTS * DIM  # 75 points × 2 dimensions = 150 doubles

# TPIE Header Constants
MAGIC = 0x521cbe927dd6056a
VERSION = 4
BLOCK_SIZE = 4096
FLAGS = 1

def create_tpie_file():
    print("=== Creating TPIE File ===")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Points: {NUM_POINTS}")
    print(f"Dimensions: {DIM}")
    print(f"Total doubles: {NUM_ITEMS}")
    print(f"Item size: {ITEM_SIZE} bytes (single double)")
    
    # Read your raw binary data
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found!")
        return
    
    with open(INPUT_FILE, "rb") as f:
        raw_data = f.read()
    
    # Check data size
    expected_size = NUM_ITEMS * ITEM_SIZE
    actual_size = len(raw_data)
    
    print(f"\nData verification:")
    print(f"  Expected: {expected_size} bytes ({NUM_ITEMS} doubles)")
    print(f"  Actual: {actual_size} bytes")
    
    if actual_size != expected_size:
        print(f"  WARNING: Size mismatch!")
        if actual_size < expected_size:
            print(f"  Your file is too small. Do you have all {NUM_POINTS} points?")
        else:
            print(f"  Your file is too large. Extra data will be ignored.")
    
    # Verify the data format by reading first few doubles
    num_doubles = min(6, actual_size // 8)
    if num_doubles > 0:
        doubles = struct.unpack(f"{num_doubles}d", raw_data[:num_doubles*8])
        print(f"\nFirst {num_doubles} doubles in file:")
        for i in range(0, num_doubles, 2):
            if i+1 < len(doubles):
                print(f"  Point {i//2}: ({doubles[i]:.6f}, {doubles[i+1]:.6f})")
    
    # Create TPIE header
    header = struct.pack(
        "QQQQQQQQQ",
        MAGIC,          # magic
        VERSION,        # version
        ITEM_SIZE,      # itemSize = 8 (one double)
        BLOCK_SIZE,     # blockSize
        0,              # userDataSize
        0,              # maxUserDataSize
        NUM_ITEMS,      # size = 150 (total number of doubles)
        FLAGS,          # flags
        0               # lastBlockReadOffset
    )
    
    # Write TPIE file
    with open(OUTPUT_FILE, "wb") as f:
        f.write(header)
        # Only write the expected amount of data
        f.write(raw_data[:expected_size])
    
    file_size = os.path.getsize(OUTPUT_FILE)
    header_size = 9 * 8  # 9 uint64_t values
    
    print(f"\n✓ Successfully created {OUTPUT_FILE}")
    print(f"  Total file size: {file_size} bytes")
    print(f"  Header size: {header_size} bytes")
    print(f"  Data size: {file_size - header_size} bytes")
    print(f"\nNow run:")
    print(f"  cd ~/learnedbench/build")
    print(f"  ./bin/bench_rsmi rsmi ../../generate_data/{OUTPUT_FILE} {NUM_POINTS} range")

if __name__ == "__main__":
    try:
        create_tpie_file()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()