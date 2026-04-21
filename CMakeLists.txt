cmake_minimum_required(VERSION 3.16)

project(learnedbench)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG")
# set(CMAKE_CXX_FLAGS "-mbmi2 -Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(PROFILE "whether starting heap profile" OFF)
option(RSMI "whether building RSMI" OFF)


set(BOOST_ROOT "/opt/homebrew/opt/boost")
set(Boost_INCLUDE_DIR "/opt/homebrew/opt/boost/include")
set(Boost_LIBRARY_DIR "/opt/homebrew/opt/boost/lib")

SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")

# check dependencies
set(TPIE_FOUND TRUE)
# Use TPIE from your local build - headers AND library from same source for consistency
set(TPIE_SOURCE_DIR "/Users/proteetiprovarawshan/tpie")
set(TPIE_BUILD_DIR "/Users/proteetiprovarawshan/tpie/build")
set(TPIE_LIBRARIES "${TPIE_BUILD_DIR}/tpie/libtpie.a")
set(TPIE_INCLUDE_DIRS "${TPIE_SOURCE_DIR}" "${TPIE_BUILD_DIR}")
# Fallback to /usr/local/lib if build folder library doesn't exist
if(NOT EXISTS ${TPIE_LIBRARIES})
  set(TPIE_LIBRARIES "/usr/local/lib/libtpie.a")
  message(STATUS "Using TPIE library from /usr/local/lib (fallback)")
else()
  message(STATUS "Using TPIE library from build folder: ${TPIE_LIBRARIES}")
endif()
include_directories(${TPIE_INCLUDE_DIRS})
#find_package(Boost 1.79.0 REQUIRED COMPONENTS program_options system filesystem)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(GEOS REQUIRED)

if(PROFILE)
  find_package(benchmark REQUIRED)
  add_definitions(-DHEAP_PROFILE)
endif()

if(RSMI)
  # Use C++17 for std::filesystem support (RSMI originally needed C++14 but torch supports C++17)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  set(TORCH_PATH "/opt/homebrew/opt/pytorch")
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
  find_package(Torch REQUIRED PATHS ${TORCH_PATH} CONFIG)

  # RSMI library path updated for your local directory
  set(RSMI_PATH "${CMAKE_SOURCE_DIR}/indexes/rsmi")

  include_directories(${Boost_INCLUDE_DIR} ${TPIE_INCLUDE_DIRS} ${RSMI_PATH})

  if(PROFILE)
    add_executable(bench_rsmi_hp bench/bench_rsmi.cpp)
    target_include_directories(bench_rsmi_hp PUBLIC "/usr/local/include" ${RSMI_PATH} ${Boost_INCLUDE_DIR})
    target_link_libraries(bench_rsmi_hp "/usr/local/lib/libtpie.a" ${TORCH_LIBRARIES} pthread "${RSMI_PATH}/build/librsmi.a" -ltcmalloc)
  else()
    # 1. CREATE the target
    add_executable(bench_rsmi bench/bench_rsmi.cpp)

    # 2. CONFIGURE the target (Absolute paths to bypass variable errors)
    target_include_directories(bench_rsmi PUBLIC
            "/Users/proteetiprovarawshan/tpie"
            "/Users/proteetiprovarawshan/tpie/build"
            "${RSMI_PATH}"
            "/opt/homebrew/opt/boost/include"
    )

    # 3. LINK the target (Explicitly use the library path we verified)
    # Note: TPIE was built with snappy compression support, so we need to link it
    # Note: Using C++17 std::filesystem since Boost filesystem not available
    target_link_libraries(bench_rsmi
            ${TPIE_LIBRARIES}
            ${TORCH_LIBRARIES}
            pthread
            "${RSMI_PATH}/build/librsmi.a"
            "-L/opt/homebrew/lib"
            "-lsnappy"
    )
  endif()

else()
  # the other benchmarks are compiled using cxx17 (required by PGM-Index)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  # ANN library path
  set(ANN_PATH "${CMAKE_SOURCE_DIR}/learnedbench/indexes/ann_1.1.2")

  include_directories(${Boost_INCLUDE_DIR} ${TPIE_INCLUDE_DIRS} ${RSMI_PATH} ${ANN_PATH})

  add_executable(datagen utils/datagen.cpp)
  target_link_libraries(datagen ${TPIE_LIBRARIES} Boost::program_options pthread)

  if(PROFILE)
    add_executable(bench2d_hp bench/bench.cpp)
    target_link_libraries(bench2d_hp ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a" -ltcmalloc)
  else()
    # add_executable(bench2d_default bench/bench.cpp)
    # target_link_libraries(bench2d_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_default PUBLIC PARTITION_NUM=100)
    
    # add_executable(bench2d_fs bench/bench.cpp)
    # target_link_libraries(bench2d_fs ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_fs PUBLIC PARTITION_NUM=60)

    # add_executable(bench2d_osm bench/bench.cpp)
    # target_link_libraries(bench2d_osm ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_osm PUBLIC PARTITION_NUM=250)

    # add_executable(bench3d_toronto bench/bench.cpp)
    # target_link_libraries(bench3d_toronto ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench3d_toronto PUBLIC BENCH_DIM=3 PARTITION_NUM=20)

    # add_executable(bench4d_default bench/bench.cpp)
    # target_link_libraries(bench4d_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench4d_default PUBLIC BENCH_DIM=4 PARTITION_NUM=10)

    # add_executable(bench6d_default bench/bench.cpp)
    # target_link_libraries(bench6d_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench6d_default PUBLIC BENCH_DIM=6 PARTITION_NUM=5)

    # add_executable(bench8d_default bench/bench.cpp)
    # target_link_libraries(bench8d_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench8d_default PUBLIC BENCH_DIM=8 PARTITION_NUM=4)

    # add_executable(bench10d_default bench/bench.cpp)
    # target_link_libraries(bench10d_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench10d_default PUBLIC BENCH_DIM=10 PARTITION_NUM=3)

    # add_executable(bench2d_1m_default bench/bench.cpp)
    # target_link_libraries(bench2d_1m_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_1m_default PUBLIC PARTITION_NUM=23)

    # add_executable(bench2d_10m_default bench/bench.cpp)
    # target_link_libraries(bench2d_10m_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_10m_default PUBLIC PARTITION_NUM=70)

    # add_executable(bench2d_50m_default bench/bench.cpp)
    # target_link_libraries(bench2d_50m_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_50m_default PUBLIC PARTITION_NUM=158)

    # add_executable(bench2d_100m_default bench/bench.cpp)
    # target_link_libraries(bench2d_100m_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_100m_default PUBLIC PARTITION_NUM=223)

    # add_executable(bench2d_e4_default bench/bench.cpp)
    # target_link_libraries(bench2d_e4_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e4_default PUBLIC PARTITION_NUM=100 INDEX_ERROR_THRESHOLD=4)

    # add_executable(bench2d_e16_default bench/bench.cpp)
    # target_link_libraries(bench2d_e16_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e16_default PUBLIC PARTITION_NUM=100 INDEX_ERROR_THRESHOLD=16)

    # add_executable(bench2d_e64_default bench/bench.cpp)
    # target_link_libraries(bench2d_e64_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e64_default PUBLIC PARTITION_NUM=100 INDEX_ERROR_THRESHOLD=64)

    # add_executable(bench2d_e256_default bench/bench.cpp)
    # target_link_libraries(bench2d_e256_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e256_default PUBLIC PARTITION_NUM=100 INDEX_ERROR_THRESHOLD=256)

    # add_executable(bench2d_e1024_default bench/bench.cpp)
    # target_link_libraries(bench2d_e1024_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e1024_default PUBLIC PARTITION_NUM=100 INDEX_ERROR_THRESHOLD=1024)

    # add_executable(bench2d_e4_fs bench/bench.cpp)
    # target_link_libraries(bench2d_e4_fs ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e4_fs PUBLIC PARTITION_NUM=60 INDEX_ERROR_THRESHOLD=4)

    # add_executable(bench2d_e16_fs bench/bench.cpp)
    # target_link_libraries(bench2d_e16_fs ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e16_fs PUBLIC PARTITION_NUM=60 INDEX_ERROR_THRESHOLD=16)

    # add_executable(bench2d_e64_fs bench/bench.cpp)
    # target_link_libraries(bench2d_e64_fs ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e64_fs PUBLIC PARTITION_NUM=60 INDEX_ERROR_THRESHOLD=64)

    # add_executable(bench2d_e256_fs bench/bench.cpp)
    # target_link_libraries(bench2d_e256_fs ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e256_fs PUBLIC PARTITION_NUM=60 INDEX_ERROR_THRESHOLD=256)

    # add_executable(bench2d_e1024_fs bench/bench.cpp)
    # target_link_libraries(bench2d_e1024_fs ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench2d_e1024_fs PUBLIC PARTITION_NUM=60 INDEX_ERROR_THRESHOLD=1024)

    # add_executable(bench3d_e4_toronto bench/bench.cpp)
    # target_link_libraries(bench3d_e4_toronto ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench3d_e4_toronto PUBLIC BENCH_DIM=3 PARTITION_NUM=20 INDEX_ERROR_THRESHOLD=4)

    # add_executable(bench3d_e16_toronto bench/bench.cpp)
    # target_link_libraries(bench3d_e16_toronto ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench3d_e16_toronto PUBLIC BENCH_DIM=3 PARTITION_NUM=20 INDEX_ERROR_THRESHOLD=16)

    # add_executable(bench3d_e64_toronto bench/bench.cpp)
    # target_link_libraries(bench3d_e64_toronto ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench3d_e64_toronto PUBLIC BENCH_DIM=3 PARTITION_NUM=20 INDEX_ERROR_THRESHOLD=64)

    # add_executable(bench3d_e256_toronto bench/bench.cpp)
    # target_link_libraries(bench3d_e256_toronto ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench3d_e256_toronto PUBLIC BENCH_DIM=3 PARTITION_NUM=20 INDEX_ERROR_THRESHOLD=256)

    # add_executable(bench3d_e1024_toronto bench/bench.cpp)
    # target_link_libraries(bench3d_e1024_toronto ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    # target_compile_definitions(bench3d_e1024_toronto PUBLIC BENCH_DIM=3 PARTITION_NUM=20 INDEX_ERROR_THRESHOLD=1024)

    add_executable(bench2d_default bench/bench.cpp)
    target_link_libraries(bench2d_default ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    target_compile_definitions(bench2d_default PUBLIC PARTITION_NUM=100)

    add_executable(bench3d_toronto bench/bench.cpp)
    target_link_libraries(bench3d_toronto ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    target_compile_definitions(bench3d_toronto PUBLIC BENCH_DIM=3 PARTITION_NUM=20)

    add_executable(bench_flood_wiki bench/bench_flood_wiki.cpp)
    target_link_libraries(bench_flood_wiki ${TPIE_LIBRARIES} Boost::program_options GEOS::geos pthread "${ANN_PATH}/lib/libANN.a")
    target_compile_options(bench_flood_wiki PRIVATE -mbmi2)

    
  endif()
endif()

# --- Data Converter Tool ---
# --- Data Converter Tool ---
add_executable(convert_data convert_data.cpp)

# 1. Tell Linker to look in Homebrew's folder
if(APPLE)
  target_link_directories(convert_data PRIVATE "/opt/homebrew/lib")
endif()

# 2. Link the libraries
target_link_libraries(convert_data
        ${TPIE_LIBRARIES}
        "snappy"      # CMake will now find this in /opt/homebrew/lib
        "pthread"
)

# 3. Include headers
target_include_directories(convert_data PRIVATE ${TPIE_INCLUDE_DIRS})