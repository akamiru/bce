**BCE v0.4.1 Release**

Copyright (C) 2016  Christoph Diegelmann

This is free software under MIT License.

BCE is a compressor for stationary data. It uses an algorithm described at http://encode.ru/threads/2150-A-new-algorithm-for-compression-using-order-n-models.

**Usage:**

    bce -c archive.bce file
     Compresses "file" to archive "archive.bce"

    bce -d file archive.bce
     Decompresses archive "archive.bce" to "file"

**Building:**

    mk build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

**Dependings:**

    libdivsufsort - See https://github.com/y-256/libdivsufsort
