#include <tpie/tpie.h>
#include <tpie/file_stream.h>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // HARDCODED PATHS - Make sure these match your system!
    std::string input_file = "/Users/proteetiprovarawshan/Thesis/generate_data/graph_data.bin";
    std::string output_file = "/Users/proteetiprovarawshan/Thesis/generate_data/graph_data_tpie.bin";

    std::cout << "Reading from: " << input_file << "\n";
    std::cout << "Writing to:   " << output_file << "\n";

    // 1. Read Raw Data
    std::ifstream in(input_file, std::ios::binary);
    if (!in) {
        std::cerr << "ERROR: Could not open input file! Check the path.\n";
        return 1;
    }

    std::vector<double> raw_values;
    double val;
    while (in.read(reinterpret_cast<char*>(&val), sizeof(double))) {
        raw_values.push_back(val);
    }
    in.close();

    if (raw_values.empty()) {
        std::cerr << "Error: No data found in input file.\n";
        return 1;
    }

    // 2. Initialize TPIE
    tpie::tpie_init();

    // 3. Write TPIE Data as DOUBLES (Item Size = 8)
    // The benchmark reads "file_stream<double>", so we must write "file_stream<double>"
    tpie::file_stream<double> out;
    out.open(output_file, tpie::access_write);

    for (double v : raw_values) {
        out.write(v);
    }

    out.close();
    tpie::tpie_finish();

    std::cout << "SUCCESS! Converted " << raw_values.size() << " doubles (Item Size: 8 bytes).\n";
    std::cout << "This represents " << raw_values.size() / 2 << " points.\n";
    return 0;
}