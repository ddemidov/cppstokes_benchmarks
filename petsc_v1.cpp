#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>
#include <cassert>

#include <petscksp.h>
#include <petsctime.h>

#include "petsc_read.hpp"

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    KSP                solver, *subksp;
    PC                 prec, subpc;
    Mat                A;
    Vec                x,f;
    PetscScalar        v;
    KSPConvergedReason reason;
    PetscInt           iters, nlocal, first_local;
    PetscReal          error;
    PetscLogDouble     tic, toc;

    char A_file[256];
    char f_file[256];
    char p_file[256];

    PetscInitialize(&argc, &argv, 0, "NS");
    PetscOptionsGetString(0, 0, "-A", A_file, 255, 0);
    PetscOptionsGetString(0, 0, "-f", f_file, 255, 0);
    PetscOptionsGetString(0, 0, "-p", p_file, 255, 0);

    read_problem(A_file, f_file, p_file, A, f, x);

    PetscTime(&tic);
    KSPCreate(MPI_COMM_WORLD, &solver);
    KSPSetType(solver, KSPBCGSL);
    KSPBCGSLSetEll(solver, 5);
    KSPSetOperators(solver,A,A);
    KSPSetTolerances(solver, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1000);

    KSPGetPC(solver,&prec);
    PCSetType(prec, PCBJACOBI);
    KSPSetFromOptions(solver);
    PCSetFromOptions(prec);
    KSPSetUp(solver);

    PCBJacobiGetSubKSP(prec, &nlocal, 0, &subksp);
    for (int i = 0; i < nlocal; ++i) {
        KSPGetPC(subksp[i], &subpc);
        PCSetType(subpc, PCILU);
        PCFactorSetLevels(subpc, 1);
    }
    PetscTime(&toc);
    double tm_setup = toc - tic;

    PetscTime(&tic);
    KSPSolve(solver,f,x);
    PetscTime(&toc);
    double tm_solve = toc - tic;

    KSPConvergedReasonView(solver, PETSC_VIEWER_STDOUT_WORLD);
    KSPGetIterationNumber(solver,&iters);
    KSPGetResidualNorm(solver,&error);
    PetscPrintf(PETSC_COMM_WORLD,"\niters: %d\nerror: %.10e", iters, error);
    PetscPrintf(PETSC_COMM_WORLD,"\nsetup: %lf\nsolve: %lf\ntotla: %lf", tm_setup, tm_solve, tm_setup+tm_solve);

    VecDestroy(&x);
    VecDestroy(&f);
    MatDestroy(&A);
    KSPDestroy(&solver);
    PetscFinalize();
}
