all: kd-tree kd-tree-iterative kd-tree-naive knn-serial

kd-tree: kd-tree.c
	gcc -o kd-tree kd-tree.c -lm

kd-tree-iterative: kd-tree-iterative.c
	gcc -o kd-tree-iterative kd-tree-iterative.c -lm	

kd-tree-naive: kd-tree-naive.cu
	nvcc kd-tree-naive.cu -o kd-tree-naive

knn-serial: knn-serial.c
	gcc -o knn-serial knn-serial.c -lm

clean:
	rm kd-tree
	rm kd-tree-iterative
	rm kd-tree-naive
	rm knn-serial

run:
	./batchrun.sh