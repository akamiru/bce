BCE v0.4 Release
Copyright (C) 2016  Christoph Diegelmann
This is free software under GNU Lesser General Public License. See <http://www.gnu.org/licenses/lgpl>

BCE is a compressor for stationary data. It uses an algorithm described at http://encode.ru/threads/2150-A-new-algorithm-for-compression-using-order-n-models.

Usage:
  bce -c archive.bce file
   Compresses "file" to archive "archive.bce"

  bce -d file archive.bce
   Decompresses archive "archive.bce" to "file"

Building:
  mk build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make
