/**
 * txt_to_tpie.cpp  —  Convert space-separated triple files to TPIE binary format
 *
 * Reads the first N lines of a space-separated text file with format:
 *   SourceID  Hop1_ID  Hop2_ID  [offset ...]
 *
 * Writes N × 3 doubles to a TPIE file_stream (interleaved x, y, z per point).
 * Compatible with bench::utils::read_points<3>() in datautils.hpp.
 *
 * Usage:
 *   csv_to_tpie <input.txt> <output.tpie> <N>
 *
 * Example:
 *   build/bin/csv_to_tpie datasets/wiki_vote_triples.txt datasets/wiki_vote_1m.tpie 1000000
 */

#include <tpie/tpie.h>
#include <tpie/file_stream.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: csv_to_tpie <input.txt> <output.tpie> <N>\n"
                  << "  Reads first N space-separated triples and writes TPIE binary.\n";
        return 1;
    }

    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    const size_t      N        = std::stoull(argv[3]);

    std::ifstream fin(in_path);
    if (!fin) {
        std::cerr << "ERROR: cannot open input file: " << in_path << "\n";
        return 1;
    }

    tpie::tpie_init();

    tpie::file_stream<double> out;
    out.open(out_path);

    std::string line;
    size_t written = 0;
    while (written < N && std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double x, y, z;
        if (!(ss >> x >> y >> z)) {
            std::cerr << "ERROR: malformed row at line " << (written + 1) << ": " << line << "\n";
            return 1;
        }
        out.write(x);
        out.write(y);
        out.write(z);
        ++written;
    }

    out.close();
    tpie::tpie_finish();

    if (written < N) {
        std::cerr << "WARNING: requested " << N << " points but file only had " << written << "\n";
    }

    std::cout << "Written " << written << " points (" << written * 3 * sizeof(double)
              << " bytes of payload) to " << out_path << "\n";
    return 0;
}
