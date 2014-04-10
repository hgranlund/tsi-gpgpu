Executables
===========

The executables for the most recent stable release is located under the folder v1.3. Every release contains two executables, Time_kd_tree_build_vx.x.exe, and Time_kd_search_vx.x.exe.


How to run
----------

#### Time_kd_tree_build_vx.x.exe:

Time_kd_tree_build generates a kd-tree and prints the timing results. The input options are described below:

__Build one tree of a give size:__

    .\time_kd_tree_build.exe <number-of-points>

So the command ```.\time_kd_tree_build.exe 100``` would bouild a kd-tree consisting of 100 random points and return the timing results for the build algorithm.

__Build several trees of increasing size:__

    .\time_kd_tree_build.exe <start number-of-points> <end number-of-points> <step>

Start number-of-points specifies the smallest tree in the series, and end number-of-points specifies the upper bound on the series. The step value determines the increase in size between each tree in the series. Every tree is built using new random point values.

So the command ```.\time_kd_tree_build.exe 100 300 150``` would build two trees, one of size 100, and one of size 250, and return the timing results for both build operations.

__Build one tree from points specified in a binary file:__

    .\time_kd_tree_build.exe <number-of-points> <path to file>

Number-of-points specifies how many points the executable will read from the data-file. This number should be lower than the actual number of points contained in the data-file. Path to file is the path to the data-file containing the points.

So the command ```.\time_kd_tree_build.exe 10000 ..\..\..\tests\data\100_mill_points.data``` would build a kd-tree from the first 10 000 points specified in the data-file.

Test data-files are located in ```/tsi-gpgpu/tests/data```.


#### Time_kd_search_vx.x.exe:

Time_kd_search generates a kd-tree, queries for all the points in the tree, and and prints the timing results. The input options are the same as for Time_kd_tree_build_vx.x.exe. Under is a short summary:

    .\time_kd_tree_build.exe <number-of-points>
    .\time_kd_tree_build.exe <start number-of-points> <end number-of-points> <step>
    .\time_kd_tree_build.exe <number-of-points> <path to file>


#### File format

The executables accepts binary files. The points should be written sequentially with x, y and z values. The methods for reading and writing points can be found at: [Github](https://github.com/hgranlund/tsi-gpgpu/blob/master/tests/kNN/kd-tree/time-kd-search.cu)


Known errors
------------

#### ErrorLaunchTimeOut:

**Reason:**

If your GPU is used for both display as well as for CUDA, then the OS will send a timeout message that will terminate the kernel/program.

**Solution:**

To solve the error one have to make the GPU only work as a Cuda GPU. There are a couple of ways to to dis:

1. (Good) Buy a separate GPU for Cuda computations.
2. (Bad) Change the timeout value in regedit. HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers --> TdrLevel = "yourTimeoutValueInSeconds".
