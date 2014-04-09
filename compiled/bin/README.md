# Executables


## How to run

#### Time_kd_tree_build_vx.x.exe:


Time_kd_tree_build generates a kd-tree and prints the timing results. The input methods are described below:

    1. .\time_kd_tree_build.exe <number-of-points>
    2. .\time_kd_tree_build.exe <start number-of-points> <end number-of-points> <step>
    2. .\time_kd_tree_build.exe <number-of-points> <path to file>

#### Time_kd_search_vx.x.exe:

Time_kd_search generates a kd-tree, query the tree with all given points, and and prints the timing results. The input methods are described below:

    1. .\time_kd_tree_build.exe <number-of-points>
    2. .\time_kd_tree_build.exe <start number-of-points> <end number-of-points> <step>
    2. .\time_kd_tree_build.exe <number-of-points> <path to file>


#### File format

The executables accepts binary files. The points should be written sequentially with x, y and z values. The methods for reading and writing points can be found at: [Github](https://github.com/hgranlund/tsi-gpgpu/blob/master/tests/kNN/kd-tree/time-kd-search.cu)




## Known errors

#### ErrorLaunchTimeOut:

**Reason:**

If your GPU is used for both display as well as for CUDA, then the OS will send a timeout message that will terminate the kernel/program.

**Solution:**

To solve the error one have to make the GPU only work as a Cuda GPU. There are a couple of ways to to dis:

1. (Good) Buy a separate GPU for Cuda computations.
2. (Bad) Change the timeout value in regedit. HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers --> TdrLevel = "yourTimeoutValueInSeconds".


