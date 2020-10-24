#ifndef PETSC_READ_HPP
#define PETSC_READ_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <cassert>

#include <amgcl/io/mm.hpp>

#include <petscksp.h>

//---------------------------------------------------------------------------
void read_problem(std::string A_file, std::string f_file, std::string p_file,
                  Mat &A, Vec &f, Vec &x, int nu = 0, IS *uis = nullptr)
{
    namespace io = amgcl::io;

    int mpi_rank;
    int mpi_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Read partition
    int n,m;
    std::vector<int> part;
    std::tie(n, m) = io::mm_reader(p_file)(part);
    assert(m == 1 && "Wrong dimensions in partitioning vector");

    if (mpi_rank == 0) {
        std::cout << "global rows: " << n << std::endl;
    }

    // Compute domain sizes
    std::vector<int> domain(mpi_size + 1, 0);
    for(auto p : part) ++domain[p+1];
    std::partial_sum(domain.begin(), domain.end(), domain.begin());

    int chunk = domain[mpi_rank+1] - domain[mpi_rank];

    if (mpi_rank == 0) {
        std::cout << "local rows:";
        for(auto d : domain) std::cout << " " << d;
        std::cout << std::endl;
    }

    // Reorder unknowns
    std::vector<int> perm(n);
    for(int i = 0; i < n; ++i) perm[i] = domain[part[i]]++;
    std::rotate(domain.begin(), domain.end()-1, domain.end());
    domain[0] = 0;

    // Read our chunk of the matrix
    std::ifstream af(A_file, std::ios::binary);
    size_t rows;
    af.read((char*)&rows, sizeof(size_t));

    assert(rows == n && "Matrix and partition dimensions differ");

    std::vector<int> ptr(chunk);
    std::vector<int> nnz(chunk);
    std::vector<int> isu;
    int uloc = 0;

    if (uis) isu.resize(chunk);

    size_t ptr_pos = af.tellg();

    ptrdiff_t glob_nnz;
    {
        af.seekg(ptr_pos + n * sizeof(ptrdiff_t));
        af.read((char*)&glob_nnz, sizeof(ptrdiff_t));

        if (mpi_rank == 0) {
            std::cout << "global nnz: " << glob_nnz << std::endl;
        }
    }

    size_t col_pos = ptr_pos + (n + 1) * sizeof(ptrdiff_t);
    size_t val_pos = col_pos + glob_nnz * sizeof(ptrdiff_t);

    for(int i = 0, j = 0; i < n; ++i) {
        if (part[i] == mpi_rank) {
            assert(perm[i] - domain[mpi_rank] == j);

            af.seekg(ptr_pos + i * sizeof(ptrdiff_t));

            ptrdiff_t p[2];
            af.read((char*)p, sizeof(p));

            ptr[j] = p[0];
            nnz[j] = p[1] - p[0];

            if (uis) {
                isu[j] = i < nu;
                uloc += isu[j];
            }

            ++j;
        }
    }

    MatCreate(MPI_COMM_WORLD, &A);
    MatSetSizes(A, chunk, chunk, n, n);
    MatSetFromOptions(A);
    MatMPIAIJSetPreallocation(A, 128, 0, 128, 0);
    MatSeqAIJSetPreallocation(A, 128, 0);

    std::vector<PetscInt>    col; col.reserve(128);
    std::vector<PetscScalar> val; val.reserve(128);

    for(int i = 0; i < chunk; ++i, col.clear(), val.clear()) {
        af.seekg(col_pos + ptr[i] * sizeof(ptrdiff_t));
        for(int j = 0; j < nnz[i]; ++j) {
            ptrdiff_t c;
            af.read((char*)&c, sizeof(ptrdiff_t));
            col.push_back(perm[c]);
        }

        af.seekg(val_pos + ptr[i] * sizeof(double));
        for(int j = 0; j < nnz[i]; ++j) {
            double v;
            af.read((char*)&v, sizeof(double));
            val.push_back(v);
        }

        PetscInt row = i + domain[mpi_rank];
        MatSetValues(A, 1, &row, col.size(), col.data(), val.data(), INSERT_VALUES);
    }

    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

    // Read our chunk of the RHS
    VecCreate(MPI_COMM_WORLD, &f);
    VecCreate(MPI_COMM_WORLD, &x);

    VecSetFromOptions(f);
    VecSetFromOptions(x);

    VecSetSizes(f, chunk, n);
    VecSetSizes(x, chunk, n);

    VecSet(x, 0.0);

    std::ifstream ff(f_file, std::ios::binary);
    {
        size_t shape[2];
        ff.read((char*)shape, sizeof(shape));

        assert(shape[0] == n);
        assert(shape[1] == 1);
    }
    size_t f_pos = ff.tellg();

    for(int i = 0, j = 0; i < n; ++i) {
        if (part[i] == mpi_rank) {
            ff.seekg(f_pos + i * sizeof(double));

            double v;
            ff.read((char*)&v, sizeof(double));

            VecSetValue(f, domain[mpi_rank] + j, v, INSERT_VALUES);
            ++j;
        }
    }

    VecAssemblyBegin(f);
    VecAssemblyEnd(f);

    if (uis) {
        std::vector<PetscInt> uidx(uloc);
        for (int i = 0, j = 0; i < chunk; ++i)
            if (isu[i]) uidx[j++] = i + domain[mpi_rank];
        ISCreateGeneral(MPI_COMM_WORLD, uloc, uidx.data(), PETSC_COPY_VALUES, uis);
    }
}

#endif
