#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

#include <petscksp.h>
#include <petsctime.h>

#include "petsc_read.hpp"

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    KSP                solver;
    PC                 prec;
    Mat                A;
    Vec                x,f;
    IS                 uis;
    PetscScalar        v;
    KSPConvergedReason reason;
    PetscInt           nu = 0, iters;
    PetscReal          error;
    PetscLogDouble     tic, toc;

    char A_file[256];
    char f_file[256];
    char p_file[256];

    PetscInitialize(&argc, &argv, 0, "NS");
    PetscOptionsGetString(0, 0, "-A", A_file, 255, 0);
    PetscOptionsGetString(0, 0, "-f", f_file, 255, 0);
    PetscOptionsGetString(0, 0, "-p", p_file, 255, 0);
    PetscOptionsGetInt(0, 0, "-u", &nu, 0);

    read_problem(A_file, f_file, p_file, A, f, x, nu, &uis);

    PetscTime(&tic);
    KSPCreate(MPI_COMM_WORLD, &solver);
    KSPSetType(solver, KSPBCGSL);
    KSPBCGSLSetEll(solver, 5);
    KSPSetOperators(solver,A,A);
    KSPSetTolerances(solver, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 500);

    KSPGetPC(solver,&prec);
    PCSetType(prec, PCFIELDSPLIT);
    PCFieldSplitSetIS(prec, "0", uis);
    PCFieldSplitSetType(prec, PC_COMPOSITE_SCHUR);
    PCFieldSplitSetSchurPre(prec, PC_FIELDSPLIT_SCHUR_PRE_SELFP, 0);
    PCFieldSplitSetSchurFactType(prec, PC_FIELDSPLIT_SCHUR_FACT_LOWER);
    PCSetUp(prec);

    PetscInt nsplits;
    KSP *subksp;
    PCFieldSplitGetSubKSP(prec, &nsplits, &subksp);
    KSPSetType(subksp[0], KSPPREONLY);
    KSPSetType(subksp[1], KSPPREONLY);
    PC pc_p, pc_u;
    KSPGetPC(subksp[0], &pc_u);
    KSPGetPC(subksp[1], &pc_p);
    PCSetType(pc_u, PCGAMG);
    PCSetType(pc_p, PCJACOBI);

    KSPSetFromOptions(solver);
    PCSetFromOptions(prec);
    KSPSetUp(solver);
    PetscFree(subksp);
    PetscTime(&toc);
    double tm_setup = toc - tic;

    PetscTime(&tic);
    KSPSolve(solver,f,x);
    PetscTime(&toc);
    double tm_solve = toc - tic;

    KSPGetIterationNumber(solver,&iters);
    KSPGetResidualNorm(solver,&error);
    PetscPrintf(PETSC_COMM_WORLD,"\niters: %d\nerror: %.10e", iters, error);
    PetscPrintf(PETSC_COMM_WORLD,"\nsetup: %lf\nsolve: %lf\ntotal: %lf\n", tm_setup, tm_solve, tm_setup+tm_solve);

    VecDestroy(&x);
    VecDestroy(&f);
    MatDestroy(&A);
    KSPDestroy(&solver);
    ISDestroy(&uis);
    PetscFinalize();
}
