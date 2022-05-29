EIGEN = /usr/include/eigen3
MPI = /user/include/mpi

test:test.cpp
	g++ -I $(EIGEN) test.cpp -o test

serial:serialNN.cpp
	g++ -I $(EIGEN) serialNN.cpp -O3 -o serial

para:para.cpp
	mpic++ -I $(EIGEN) para.cpp -O3 -o para
compare:compare.cpp
	g++ -I $(EIGEN) compare.cpp -O3 -o compare