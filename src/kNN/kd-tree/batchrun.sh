#/bin/bash
echo "Testing kd-tree..."
for n in 10000 100000 1000000 #10000000
do
    ./kd-tree $n
done
echo "Done!\n"

echo "Testing kd-tree-iterative..."
for n in 10000 #100000 1000000 #10000000
do
    ./kd-tree-iterative #$n
done
echo "Done!\n"

echo "Testing kd-tree-naive..."
for n in 10000 #100000 1000000 #10000000
do
    ./kd-tree-naive #$n
done
echo "Done!\n"

echo "Testing knn-serial..."
for n in 10000 100000 1000000 #10000000
do
    ./knn-serial $n
done
echo "Done!\n"

# echo "Testing knn-serial..."
# echo "serial timing results (ms):" > knn-serial-test-data.dat
# for n in 10000 100000 1000000 10000000
# do
#     echo "with" $n "random data points..."
#    ./knn-serial $n >> knn-serial-test-data.dat
# done
# echo "Done!"
