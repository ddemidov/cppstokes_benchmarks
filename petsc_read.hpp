#ifndef PETSC_READ_HPP
#define PETSC_READ_HPP

#include <string>

#include <petscmat.h>
#include <petscvec.h>
#include <petscistypes.h>

//---------------------------------------------------------------------------
void read_problem(std::string A_file, std::string f_file, std::string p_file,
                  Mat &A, Vec &f, Vec &x, int nu = 0, IS *uis = nullptr);

#endif
