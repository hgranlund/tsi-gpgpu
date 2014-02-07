all: knn-serial bitonic-sort-serial knn-brute-force knn-garcia

knn-brute-force: kNN-brute-force.cu
	nvcc kNN-brute-force.cu -o kNN-brute-force -lcuda

knn-garcia: kNN-brute-force-garcia.cu
	nvcc kNN-brute-force-garcia.cu -o kNN-garcia -lcuda

bitonic-sort-serial: bitonic-sort-serial.c
	cl -o bitonic-sort-serial bitonic-sort-serial.c

# Still needs gcc to compile due to compatibility problems
knn-serial: knn-serial.c
	gcc -o knn-serial knn-serial.c -lm

clean:
	rm *.exe
	rm *.obj
	rm *.exp
	rm *.lib

run-win:
	.\kNN-brute-force.exe
	.\kNN-garcia.exe
	.\bitonic-sort-serial.exe
	.\knn-serial.exe