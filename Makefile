NVCC=nvcc -g -G -w -arch=sm_60 -std=c++11 -pg -L/usr/local/cuda-9.0/lib64 -lcublas -O3 -Xcompiler -fopenmp -I ./include -I /usr/local/cuda-9.0/include/  

all: knnjoin
knnjoin: ./src/knnjoin.cu
	$(NVCC) -o $@ $^
clean:
	rm knnjoin *.out
