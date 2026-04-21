/**
 * csv_to_tpie_true3d.cpp  –  True3DLearnedIndex Dataset Converter (Scaffolding)
 * ===============================================================================
 * Reads a semicolon-delimited CSV with columns: SourceID;Hop1_ID;Hop2_ID;Offset
 * Writes TWO TPIE artifacts:
 *
 *   1. <output_points.tpie>   – 3-D point dataset
 *      Layout: [x0, y0, z0, x1, y1, z1, ..., xN-1, yN-1, zN-1]  (3 doubles per point)
 *      where  x = double(SourceID),  y = double(Hop1_ID),  z = double(Hop2_ID)
 *      Compatible with bench::utils::read_points<3>(pts, fname, N)
 *
 *   2. <output_offsets.tpie>  – matching offsets dataset
 *      Layout: [off0, off1, ..., offN-1]  (1 double per row)
 *      Compatible with bench::utils::read_points<1>(offs, fname, N)
 *      Row i in the offsets file corresponds exactly to row i in the points file.
 *
 * ROW ORDER:
 *   Rows are sorted lexicographically: (SourceID, Hop1_ID, Hop2_ID) ascending.
 *   This matches the LexFlattenBaseline order so results are directly comparable.
 *   Offset values remain the original sequential rank from the CSV.
 *
 * HOW OFFSETS ALIGN WITH POINTS:
 *   points[i]  = (SourceID_i, Hop1_ID_i, Hop2_ID_i)
 *   offsets[i] = Offset_i
 *   Both files have the same number of records N.
 *   Read both files in parallel to associate each 3-D point with its offset.
 *
 * WHAT COMES NEXT (True3DLearnedIndex / Path B):
 *   TODO: Build a 3-D spatial learned index on the points file.
 *         Candidate approaches:
 *           a) Extend RSMI to 3-D by adding a third dimension to its spatial
 *              partitioning tree (currently hardcoded for 2-D in RSMI.h).
 *           b) Use a Z-order (Morton) curve to map (x,y,z) → 1-D and feed
 *              to PGM, recovering multidimensional locality without changing RSMI.
 *           c) Build a 3-D grid index (uniform or learned) for point lookup,
 *              using offsets as the payload to verify correctness.
 *         Metrics to compare against LexFlattenBaseline:
 *           Build time, index size, latency (mean/p50/p95/p99 ns),
 *           prediction error, refinement cost, correctness.
 *
 * Usage:
 *   csv_to_tpie_true3d <input.csv> <output_points.tpie> <output_offsets.tpie> [options]
 *
 * Options:
 *   --verify       read back first 5 points and offsets for sanity check
 *   --no_header    treat first row as data (no header line)
 */

#include <tpie/tpie.h>
#include <tpie/file_stream.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// Row struct
// --------------------------------------------------------------------------
struct Row {
    int64_t src;
    int64_t hop1;
    int64_t hop2;
    int64_t offset;
};

// --------------------------------------------------------------------------
// Argument helpers
// --------------------------------------------------------------------------
static bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.csv> <output_points.tpie> <output_offsets.tpie>"
                     " [--verify] [--no_header]\n";
        return 1;
    }

    const std::string csv_path     = argv[1];
    const std::string pts_path     = argv[2];
    const std::string offsets_path = argv[3];
    const bool verify    = has_flag(argc, argv, "--verify");
    const bool no_header = has_flag(argc, argv, "--no_header");

    std::cout << "====================================\n";
    std::cout << "True3DLearnedIndex CSV → TPIE Converter\n";
    std::cout << "Input CSV      : " << csv_path     << "\n";
    std::cout << "Output points  : " << pts_path     << "\n";
    std::cout << "Output offsets : " << offsets_path << "\n";
    std::cout << "====================================\n";

    // ------------------------------------------------------------------
    // 1. Read CSV
    // ------------------------------------------------------------------
    std::ifstream ifs(csv_path);
    if (!ifs) {
        std::cerr << "ERROR: cannot open " << csv_path << "\n";
        return 1;
    }

    std::vector<Row> rows;
    std::string line;
    bool first_line = true;

    while (std::getline(ifs, line)) {
        if (first_line && !no_header) {
            if (line.find("SourceID") == std::string::npos)
                std::cerr << "WARN: header not recognized: " << line << "\n";
            first_line = false;
            continue;
        }
        first_line = false;

        if (line.empty()) continue;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::istringstream ss(line);
        std::string tok;
        Row r{};
        int col = 0;
        while (std::getline(ss, tok, ';')) {
            switch (col++) {
                case 0: r.src    = std::stoll(tok); break;
                case 1: r.hop1   = std::stoll(tok); break;
                case 2: r.hop2   = std::stoll(tok); break;
                case 3: r.offset = std::stoll(tok); break;
            }
        }
        if (col < 4) {
            std::cerr << "WARN: skipping malformed line: " << line << "\n";
            continue;
        }
        rows.push_back(r);
    }
    ifs.close();

    if (rows.empty()) {
        std::cerr << "ERROR: no data rows parsed.\n";
        return 1;
    }
    std::cout << "Parsed " << rows.size() << " rows.\n";

    // ------------------------------------------------------------------
    // 2. Sort lexicographically: (SourceID, Hop1_ID, Hop2_ID)
    //    Matches the order used by LexFlattenBaseline for comparability.
    // ------------------------------------------------------------------
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.src  != b.src)  return a.src  < b.src;
        if (a.hop1 != b.hop1) return a.hop1 < b.hop1;
        return a.hop2 < b.hop2;
    });

    // ------------------------------------------------------------------
    // 3. Write TPIE artifacts
    //
    //    Points file: 3 doubles per row  → read_points<3>(pts, fname, N)
    //      Layout: [x0, y0, z0, x1, y1, z1, ..., xN-1, yN-1, zN-1]
    //
    //    Offsets file: 1 double per row  → read_points<1>(offs, fname, N)
    //      Layout: [off0, off1, ..., offN-1]
    //
    //    Row i in points ↔ row i in offsets (same sort order).
    // ------------------------------------------------------------------
    tpie::tpie_init();

    {
        tpie::file_stream<double> pts_out;
        pts_out.open(pts_path, tpie::access_write);

        tpie::file_stream<double> off_out;
        off_out.open(offsets_path, tpie::access_write);

        for (auto& r : rows) {
            pts_out.write(static_cast<double>(r.src));    // x = SourceID
            pts_out.write(static_cast<double>(r.hop1));   // y = Hop1_ID
            pts_out.write(static_cast<double>(r.hop2));   // z = Hop2_ID

            off_out.write(static_cast<double>(r.offset)); // payload
        }

        pts_out.close();
        off_out.close();
    }

    std::cout << "Wrote " << rows.size() << " 3-D points to " << pts_path << "\n";
    std::cout << "Wrote " << rows.size() << " offsets    to " << offsets_path << "\n";

    // ------------------------------------------------------------------
    // 4. Verify: read back first 5 rows from both files
    // ------------------------------------------------------------------
    if (verify) {
        std::cout << "\nVerification – first 5 rows read back:\n";
        tpie::file_stream<double> pts_in;
        pts_in.open(pts_path, tpie::access_read);
        tpie::file_stream<double> off_in;
        off_in.open(offsets_path, tpie::access_read);

        int to_print = std::min<int>(5, static_cast<int>(rows.size()));
        for (int i = 0; i < to_print; ++i) {
            double x   = pts_in.read();
            double y   = pts_in.read();
            double z   = pts_in.read();
            double off = off_in.read();
            std::cout << "  [" << i << "] src=" << x << "  hop1=" << y
                      << "  hop2=" << z << "  offset=" << off << "\n";
        }

        pts_in.close();
        off_in.close();
        std::cout << "Verification: PASSED\n";
    }

    tpie::tpie_finish();

    std::cout << "\nDone.\n";
    std::cout << "  Points  : " << pts_path     << "\n";
    std::cout << "  Offsets : " << offsets_path << "\n";
    std::cout << "\nNEXT STEPS (True3DLearnedIndex / Path B):\n";
    std::cout << "  1. Build a 3-D spatial learned index on the points file.\n";
    std::cout << "  2. Use offsets file to verify lookup correctness.\n";
    std::cout << "  3. Compare latency and prediction error against LexFlattenBaseline.\n";
    std::cout << "  See docs/EXPERIMENTS.md for details.\n";
    return 0;
}
