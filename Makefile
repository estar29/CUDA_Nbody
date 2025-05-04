CXXFLAGS=-O3

# Sequential implementation.
nbody_seq: nbody_seq.cpp
	g++ -O3 nbody_seq.cpp -o nbody_seq

solar_seq.out: nbody_seq
	date
	./nbody_seq planet 200 5000000 10000 > solar_seq.out # maybe a minutes
	date

solar_seq.pdf: solar_seq.out
	python3 plot.py solar_seq.out solar_seq.pdf 1000 

random_seq.out: nbody_seq
	date
	./nbody_seq 1000 1 10000 100 > random_seq.out # maybe 5 minutes
	date

# CUDA implementation.
#nbody_cuda : nbody_cuda.cu
#	nvcc

#solar_cuda.out: nbody_cuda
#	date
#	./nbody_cuda planet 200 5000000 10000 > solar_cuda.out # Should be faster!
#	date

#solar_cuda.pdf : solar_cuda.out
#	python2 plot.py solar_cuda.out solar_cuda.pdf 1000

#random_cuda.out : nbody_cuda
#	date
#	./nbody_cuda 1000 1 10000 100 > random_cuda.out
#	date
