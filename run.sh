#!/bin/bash
set -e

dset="https://zenodo.org/record/4134357/files"
np=8

lscpu

for p in $( ./problems.py ); do
    echo
    echo --- ${p} ---
    A=${p}_A.bin
    b=${p}_b.bin
    u=${p}_u.txt
    wget ${dset}/${A} ${dset}/${b} ${dset}/${u}
    partition -B -i ${A} -o part.bin -n ${np}
    echo
    echo --- PETSC ---
    OMP_NUM_THREADS=1 mpirun -np ${np} petsc_fs -A ${A} -f ${b}-u $( cat ${u} ) -p part.bin  -ksp_monitor -memory_view
    echo
    echo --- AMGCL ---
    OMP_NUM_THREADS=${np} OMP_PLACES=cores amgcl_spc_block_mixed -A ${A} -f ${b} -u $( cat ${u} )
    rm ${A} ${b} ${u} part.bin
