**BCE v0.4.1 Release**

Copyright (C) 2016  Christoph Diegelmann

This is free software under MIT License. See https://opensource.org/licenses/MIT.

BCE is a compressor for stationary data. It uses an algorithm described at http://encode.ru/threads/2150-A-new-algorithm-for-compression-using-order-n-models.

The algorithm is also known as Compression by Substring Enumeration (CSE) in literature.

**Usage:**

    bce -c archive.bce file
     Compresses "file" to archive "archive.bce"

    bce -d file archive.bce
     Decompresses archive "archive.bce" to "file"

**Building:**

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

**Dependings:**

  libdivsufsort - See https://github.com/akamiru/libdivsufsort
