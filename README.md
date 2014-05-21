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
