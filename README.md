# BCE
BCE v0.5 AVX2 branch - compressor for stationary data

## Compile with

    g++ -std=c++14 -O3 -march=native -o bce5 main.cpp
    
## Run with
Unzip enwik8.zip for enwik8.bwt than running it will report a time.

    ./bce5
    
## Results so far

    i7 4770K    16 GB dual   channel DDR 3    ~5,75 s
    E5 1630 v4  64 GB quad   channel DDR 4    ~7,70 s
    i7 6700      8 GB single channel DDR 4    ~8,20 s
