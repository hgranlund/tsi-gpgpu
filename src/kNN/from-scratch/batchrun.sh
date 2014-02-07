#/bin/bash

echo "Testing knn-serial..."
echo "serial timing results (ms):" > knn-serial-test-data.dat
for n in 10000 100000 1000000 10000000
do
    echo "with" $n "random data points..."
   ./knn-serial $n >> knn-serial-test-data.dat
done
echo "Done!"