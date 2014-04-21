Analysis of options for acquiring GPGPU capabilities at IPM
===========================================================

This document is an quick analysis of different possibilities for acquiring GPGPU hardware capabilities for use in research at IPM, NTNU. Three main options, and their main benefits and drawbacks, is detailed under.


Buying GPGPU hardware
---------------------

__Benefits__

* When the hardware is bought, you have unlimited access to GPGPU hardware.

__Drawbacks__

* High initial price.
* IPM is responsible for maintaining and upgrading the hardware.

This can be achieved in several ways, depending on existing hardware at IPM, and the level of support wanted for the end product.

### Buying only a GPU-card for CUDA GPGPU operations, and combining it with an existing computer.

This might be possible for research projects that already have access to a decent desktop computer, and just need to add a suitable graphics card to be able to run heavy calculations. This is indeed the case with our master thesis, but might not be the case for other master students or other users. You also does not have the possibility to share the computing resource.

Popular GPUs for GPGPU computing is the nVidia Tesla K20 and K40 cards. Although K40 is the definitively best cards of the two, they both have excellent performance, is CUDA compatible and makes separating rendering from GPU computing possible under Windows.

[nVidia Tesla K40](https://www.komplett.no/pny-nvidia-tesla-k40-workstation-card/812833) can be bought from Komplett.no for 51000 and [nVidia Tesla K20](https://www.komplett.no/pny-nvidia-tesla-k20-workstation-card/769725) can be bought for NOK 32000 NOK.

### Building a custom desktop HPC (high performance computer)

In order to eliminate the possible problem of researchers not already having access to a suitable desktop computer, a custom desktop computer could be built around one of the Tesla cards. nVidia has written up a relevant [Guide to Building Your Own Tesla Personal Supercomputer System](http://www.nvidia.com/object/tesla_build_your_own.html). The entire computer could then be loaned out to researchers in need of this kind of hardware.

An additional 20000 NOK should be factored in when considering building such a dedicated computer, as well as the risk and reliability problems associated with building and running a custom HPC desktop computer.

### Buying a pre-built desktop or server solution

Buying a pre-built desktop or server HPC solution from one of the [nVidia Tesla resellers](http://www.nvidia.com/object/where-to-buy-tesla.html) is the option with the lowest risk involved. You are guarantied a working system, and will have a vendor for support if something decides to go wrong. Choosing a server based product could also give several researchers access to the hardware remotely at the same time.

This is a quite pricey option, and you usually have to ask for a quote on price, but I would expect prices to exceed at least 80000 - 100000 NOK for a single system.


Relying on a cloud service
--------------------------

__Benefits__

* Low initial price.
* Maintenance and upgrades is included.
* Scales according to demand, and you only pay for what you use.

__Drawbacks__

* Can be pricey in the long run.
* Security issues when using a cloud service in combination with confidential data.

We have experimented with using [amazon web services](https://aws.amazon.com/hpc/) and have had no problems using this service. The cost is ~5 NOK per hour of computing time.


Borrowing or renting GPGPU hardware from another faculty at NTNU
----------------------------------------------------------------

A bit of searching and inquiring lead us to investigate the [IDI/NTNU HPC-LAB](http://research.idi.ntnu.no/hpc-lab/)  directed by Dr. Anne C. Elster.

__Benefits__

* No cost, or some reasonable loan fee from IDI.
* Lots of high end graphics cards.
* Maintenance and upgrades is performed by the staff at IDI HPC-LAB.
* A cooperation with IDI HPC-LAB might result in beneficial knowledge transfer.

__Drawbacks__

* Possible NTNU bureaucracy and politics.

A email has been sent to Anne Elster regarding the possibility of using the HPC-LAB.