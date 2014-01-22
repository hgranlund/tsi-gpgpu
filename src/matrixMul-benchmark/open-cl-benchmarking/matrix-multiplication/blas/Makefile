CC = g++


#---------------------------------------------------------------------------
#  Set up location of OpenCL sdk library and runtime installed location
#---------------------------------------------------------------------------

CLLIB  = -L/usr/local/clAmdBlas-1.4/lib64/ -L/usr/local/ati-stream-sdk-v2.3-lnx64/lib/x86_64/
CLINC = -I/usr/local/ati-stream-sdk-v2.3-lnx64/include/ -I/usr/local/clAmdBlas-1.4/include
CLLINKFLAGS=-lOpenCL -lclAmdBlas


# Name of set of example program
OBJECTS = AMD-APP-OpenclDgemmGPU


$(OBJECTS):%:%.c
	$(CC) $(CLLINKFLAGS) $(CLLIB) $(CLINC) -o $@ $@.c
clean :
	rm -rf *.o $(OBJECTS)
