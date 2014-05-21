tsi-gpgpu
=========

A project to investigate the power of GPGPU programming for engineering tasks.


The quick n'dirty guide to getting started on Windows
-----------------------------------------------------

Requirements:

* Microsoft Visual Studio 2010
* CUDA 5.5
* Git
* CMake

To get up and running, open the Visual Studio Command Prompt, or any other shell where the Visual Studio tool-chain is set up, and do the following:

1) Clone the source from GitHub.

    git clone https://github.com/hgranlund/tsi-gpgpu.git
    
2) Make a build folder and build the project with CMake.

    $> cd "to project root"\tsi-gpgpu
    ..\tsi-gpu> mkdir build $$ cd build
    # cmake -G "Visual Studio 10.0" ..\
    ..\tsi-gpu\build> cmake -G "NMake Makefiles" ..\
    ..\tsi-gpu\build> nmake
    
3) Check that all is good and run the current tests.

    $> cd "to build folder"
    ..\tsi-gpu\build> nmake test

All executables will be in ```/build/bin``` and all libraries will be in ```/build/lib/```.


Folder layout
-------------

Code and documentation for different experiments are placed under the ```/src``` folder. Some source folders are supplies with informative README.md files. Visit [tsi-gpgpu/src/kNN](https://github.com/hgranlund/tsi-gpgpu/tree/master/src/kNN) for the most recent update on the kNN effort.

Relevant papers and other background material is placed in the ```/resources``` folder.


Installation notes for Windows
------------------------------

#### Installing CUDA

Get the installer at the [download page](https://developer.nvidia.com/cuda-downloads). NVIDIA has a good [getting started guide](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html).

#### Installing Git

GitHub supplies an [native app for Windows](https://help.github.com/articles/set-up-git). This is the easy way to get started. Remember to check the options to add Git to your environment variables, if you want to be able to use Git from CMD or PowerShell. If this is problematic, you can also use the GUI for getting the source code.

#### Installing CMake

Visit [CMake.org](http://www.cmake.org/cmake/resources/software.html) and find a suitable installer for your system.

#### Bonus snacks for *NIX addicts

* Virtual desktops can be added to Windows with the [Desktops](http://technet.microsoft.com/en-us/sysinternals/cc817881.aspx) add-on.
* PowerShell is your new bash.
* [PsGet](http://psget.net/) is a package manager for PowerShell modules. Try it out with [Posh-Git](http://www.imtraum.com/blog/streamline-git-with-powershell/).
* [PowerShell Community Extensions](http://pscx.codeplex.com/) has a lot of good stuff.
* Set up Powershell to work as your Visual Studio Command Prompt by loading vsvars.bat.


Installation notes for Ubuntu
-----------------------------

#### Installing CUDA

    sudo apt-get install nvidia-cuda-toolkit
    
#### Installing Git

    apt-get install git

#### Installing CMake

    sudo apt-get install cmake

#### Build with

    ...\tsi-gpu> mkdir build $$ cd build
    ...\tsi-gpu/build> cmake ../
    ...\tsi-gpu/build> make

All executables will be in ```/build/bin``` and all libraries will be in ```/build/lib/```.


Timing executables
------------------

The project comes with three timing test executables, time_kd_search, time_kd_search_openmp and time_kd_tree build. The following section describes how to use these executables for timing tests of the different library functions. Time_kd_tree build is documented in more detail as a reference for anyone not that familiar with a command line.


#### Time_kd_search:

Time_kd_search generates a kd-tree, queries for all the points in the tree, or a specified number of points, and then prints the timing results. The calculation is performed using the GPU, and the API function cuQueryAll. Under is a short summary of the different input options:

    .\time_kd_search.exe <number-of-points>
    .\time_kd_search.exe <number-of-points> <path to file>
    .\time_kd_search.exe <start number-of-points> <end number-of-points> <step>
    .\time_kd_search.exe <start number-of-points> <end number-of-points> <step> <number-of-k>
    .\time_kd_search.exe <start number-of-points> <end number-of-points> <step> <number-of-query-points>


#### Time_kd_search_openmp:

Time_kd_search_openmp generates a kd-tree, queries for all the points in the tree, or a specified number of points, and then prints the timing results. The calculation is performed using the CPU, and the API function mpQueryAll. Under is a short summary of the different input options:

    .\time_kd_search_openmp.exe <number-of-points>
    .\time_kd_search_openmp.exe <number-of-points> <path to file>
    .\time_kd_search_openmp.exe <start number-of-points> <end number-of-points> <step>
    .\time_kd_search_openmp.exe <start number-of-points> <end number-of-points> <step> <number-of-k>
    .\time_kd_search_openmp.exe <start number-of-points> <end number-of-points> <step> <number-of-query-points>


#### Time_kd_tree_build:

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

So the command ```.\time_kd_tree_build.exe 10000 \path-to-file\100_mill_points.data``` would build a kd-tree from the first 10 000 points specified in the data-file.


#### File format

The executables accepts binary files. The points should be written sequentially with x, y and z values. The methods for reading and writing points can be found at: [Github](https://github.com/hgranlund/tsi-gpgpu/blob/master/tests/kNN/kd-tree/time-kd-search.cu)


Known errors
------------

#### ErrorLaunchTimeOut:

**Reason:**

If your GPU is used for both display as well as for CUDA, then the OS will send a timeout message that will terminate the kernel/program.

**Solution:**

To solve the error one have to make the GPU only work as a CUDA GPU. There are a couple of ways to to dis:

1. (Good) Buy a separate GPU for CUDA computations.
2. (Bad) Change the timeout value in regedit. HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers --> TdrLevel = "yourTimeoutValueInSeconds".
