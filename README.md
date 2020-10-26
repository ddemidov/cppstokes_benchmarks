Source code accompanying the manuscript

"Accelerating linear solvers for Stokes problems with C++ metaprogramming"
by Denis Demidov, Lin Mu, Bin Wang

The system matrices and the RHS vectors used for the benchmarks may be retrieved here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4134357.svg)](https://doi.org/10.5281/zenodo.4134357)

The files in the dataset correspond to the Stokes equation discretized for 3 different cases:

* Unit cube problem (`ucube`). A rotating flow driven by an external force f in a closed unit cube.
* Converging-diverging tube problem (`cdtube`). Pressure-driven tube flow through a 3D converging-diverging tube under a pressure drop of 1Pa.
* Sphere packing problem (`spack`). A complex sphere packing flow problem with non-uniform cell size distribution and large cell size contrast.

Each problem contains the system matrix (`A.bin`), the RHS vector (`b.bin`),
and the text file containing the number of DOFs corresponding to the velocity
field (`u.txt`) for 6 to 7 different problem sizes.

In order to run the PETSc benchmarks, the systems need to be partitioned. This
can be done with the provided `partition.cpp`. The following example partitions
the matrix for 4 MPI processes:

```sh
$ ./partition -B -i A.bin -n 4 -o part4.bin
```

An example of running a PETSc benchmark:
```sh
$ export OMP_NUM_THREADS=1
$ mpirun -np 4 ./petsc_v1 -A A.bin -f b.bin -p part4.bin
$ mpirun -np 4 ./petsc_fs -A A.bin -f b.bin -p part4.bin -u $(cat u.txt)
```

An example of running an AMGCL benchmark:
```sh
./amgcl_spc_block_mixed -A A.bin -f b.bin -u $(cat u.txt)
```