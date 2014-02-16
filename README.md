tsi-gpgpu
=========

A project to investigate the power of GPGPU programming for engineering tasks.

Code and documentation for different experiments are placed under the ```/src``` folder.


The quick n'dirty guide to getting started on Windows
-----------------------------------------------------

Currently, cl and cmake/nmake is used to build and compile the C code. If you have a Windows system with these commands available in a command-line environment, you're good to go. Finally you'll need a suitable installation of CUDA.


### Installing CUDA

Get the installer at the [download page](https://developer.nvidia.com/cuda-downloads). NVIDIA has a good [getting started guide](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html).


### Bonus snacks for *NIX addicts

* Virtual desktops can be used with the [Desktops](http://technet.microsoft.com/en-us/sysinternals/cc817881.aspx) add-on in Windows.
* PowerShell is your new bash.
* [PsGet](http://psget.net/) is a package manager for PowerShell modules. Try it out with [Posh-Git](http://www.imtraum.com/blog/streamline-git-with-powershell/).


### Build project

* Use Visual Studio Command propt (2010).
* Install cmake.
* Install git.
* Install Cuda Toolkit.

Clone git project:

    git clone https://github.com/hgranlund/tsi-gpgpu.git
    
Build project:

    cd "to project root"\tsi-gpgpu
    mkdir build $$ cd build
    cmake -G "NMake Makefiles ..\
    nmake
    
    
Run correctness tests:

    cd "to build folder"
    nmake test

All executables will be in ***/build/bin***.

All libraries will be in ***/build/lib/***.

## Staring guide on Ubuntu

### Install requirements

 **CUDA toolkit:**

    sudo apt-get install nvidia-cuda-toolkit

 **cmake:**

    sudo apt-get install cmake

### Build

    mkdir build $$ cd build
    cmake ../
    make

All executables will be in ***/build/bin***.

All libraries will be in ***/build/lib/***.
