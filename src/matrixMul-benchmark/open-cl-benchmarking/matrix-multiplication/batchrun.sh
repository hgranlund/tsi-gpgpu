#/bin/bash

echo "Testing mult-matrix..."
echo "serial timing results (ms):" > mult-matrix-test-data.dat
for n in 128 256 512 1024
do
   ./mult-matrix $n >> mult-matrix-test-data.dat
done
echo "Done!"

echo "blas timing results (ms):" > blas-mult-matrix-test-data.dat
echo "Testing blas-mult-matrix..."
for n in 128 256 512 1024 2048 4096
do
   ./blas-mult-matrix $n >> blas-mult-matrix-test-data.dat
done
echo "Done!"

echo "opencl timing results (ms):" > opencl-mult-matrix-test-data.dat
echo "Testing opencl-mult-matrix..."
for n in 128 256 512 1024 2048 4096
do
   ./opencl-mult-matrix $n >> opencl-mult-matrix-test-data.dat
done
echo "Done!"