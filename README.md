tsi-gpgpu
=========

A project to investigate the power of GPGPU programming for engineering tasks.


## The quick n'dirty guide to getting started on Windows

Currently, gcc and make is used to build and compile the C code. If you have a Windows system with these commands available in a command-line environment, you're good to go. Finally you'll need a suitable installation of CUDA.

Alternatively, you should be able to use cl and nmake from the VisualStudio command-line toolset, but this option has not been tested. More information about this topic is available at [msdn](http://msdn.microsoft.com/en-us/library/f35ctcxw.aspx). You'll probably need to make some changes to the makefiles in order to make them use the cl compiler.

To acquire a version of gcc and make on Windows, several options are available. A quick guide using MinGW gcc and GNUMake for windows is outlined under. Another popular option is to use [Cygwin](http://www.cygwin.com/).


### Installing gcc with MinGW

Get hold of the MinGW installer from [sourceforge](http://sourceforge.net/projects/mingw-w64/files/latest/download?source=files). The link should take you directly to the latest version and more options can be found [here](http://sourceforge.net/apps/trac/mingw-w64/wiki/GeneralUsageInstructions).

Run the installer and make a note of the install-folder. Then add the ```your-mingw-install-folder\mingw64\bin``` folder to your path environment variable. This folder contains the gcc executable, so you should now be able to running ```$> gcc --version``` in CMD or PowerShell.


### Installing GNUMake

Again get hold of the installer at [sourceforge](http://sourceforge.net/projects/gnuwin32/files/make/3.81/make-3.81.exe/download?use_mirror=dfn&download=). Other versions are available at the [GNU make](http://gnuwin32.sourceforge.net/packages/make.htm) site.

Run the installer [GNU make](http://gnuwin32.sourceforge.net/packages/make.htm), and add the ```your-gnumake-install-folder\mingw64\bin``` folder to your path environment variable. You can test the installation by running ```$> make --version``` in CMD or ProwerShell.


### Installing CUDA

Get the installer at the [download page](https://developer.nvidia.com/cuda-downloads). NVIDIA has a good [getting started guide](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html).