all: kd-tree knn-serial

kd-tree: kd-tree.c
	gcc -o kd-tree kd-tree.c

knn-serial: knn-serial.c
	gcc -o knn-serial knn-serial.c -lm

clean:
	rm kd-tree
	rm knn-serial

run:
	./kd-tree
	./knn-serial