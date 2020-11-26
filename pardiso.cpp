#include <iostream>
#include <fstream>
#include <string>

#include <boost/program_options.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/PardisoSupport>

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "show help")
        ("matrix,A", po::value<std::string>()->required(), "The system matrix")
        ("rhs,f", po::value<std::string>()->required(),    "The right-hand side")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    // Read the system
    ptrdiff_t rows;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs;

    {
        std::ifstream f(vm["matrix"].as<std::string>(), std::ios::binary);
        f.read((char*)&rows, sizeof(ptrdiff_t));
        ptr.resize(rows + 1);
        f.read((char*)ptr.data(), ptr.size() * sizeof(ptrdiff_t));
        col.resize(ptr.back());
        val.resize(ptr.back());
        f.read((char*)col.data(), col.size() * sizeof(ptrdiff_t));
        f.read((char*)val.data(), val.size() * sizeof(double));
    }

    {
        std::ifstream f(vm["rhs"].as<std::string>(), std::ios::binary);
        rhs.resize(rows);
        f.read((char*)rhs.data(), rhs.size() * sizeof(double));
    }

    // Map the vectors to Eigen datatypes:
    Eigen::MappedSparseMatrix<double, Eigen::RowMajor, ptrdiff_t> A(rows, rows, ptr.back(), ptr.data(), col.data(), val.data());
    Eigen::Map<Eigen::VectorXd> b(rhs.data(), rhs.size());

    // Setup the Pardiso solver
    Eigen::PardisoLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor, int>> solver;
    solver.pardisoParameterArray()[9]  = 8; // Pivoting perturbation.
    solver.pardisoParameterArray()[10] = 0; // Use nonsymmetric permutation and scaling MPS
    solver.pardisoParameterArray()[12] = 1; // Improved accuracy using (non-) symmetric weighted matching.
    solver.compute(A);                      // Factorize A

    //Solve the system
    Eigen::VectorXd x(rows);
    x = solver.solve(b);

    double pardiso_memory1 = solver.pardisoParameterArray()[14];
    double pardiso_memory2 = solver.pardisoParameterArray()[15] + solver.pardisoParameterArray()[16];
    double peak_memory = std::max(pardiso_memory1, pardiso_memory2);

    std::cout
        << "Perturbed pivots: " << solver.pardisoParameterArray()[13] << std::endl
        << "Nonzeros in the factor LU: " << solver.pardisoParameterArray()[17] << std::endl
        << "MFlops for LU factorization: " << solver.pardisoParameterArray()[18] << std::endl
        << "Solver memory usage: " << peak_memory * 1024 << std::endl
        << "Solution successful: " << std::boolalpha << (solver.info() == 0) << std::endl;
}
