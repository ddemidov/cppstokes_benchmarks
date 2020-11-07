#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/idrs.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/iluk.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
template <class Matrix>
void solve(const Matrix &K, const std::vector<double> &rhs)
{
    auto t1 = prof.scoped_tic("amgcl");

    prof.tic("setup");
    typedef amgcl::backend::builtin<double> Backend;
    typedef amgcl::make_solver<
        amgcl::relaxation::as_preconditioner<
            Backend, amgcl::relaxation::iluk>,
        amgcl::solver::idrs<Backend>
        > Solver;

    Solver::params prm;
    prm.solver.s = 5;
    prm.solver.maxiter=2000;
    prm.solver.tol = 1e-12;
    prm.solver.replacement = true;
    prm.solver.smoothing = true;

    Solver solve(K, prm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    auto n = amgcl::backend::rows(K);

    amgcl::backend::numa_vector<double> f(rhs);
    amgcl::backend::numa_vector<double> x(n);

    size_t iters;
    double error;

    prof.tic("solve");
    std::tie(iters, error) = solve(K, f, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;

    amgcl::backend::numa_vector<double> r(n);
    amgcl::backend::residual(f, K, x, r);
    std::cout << "True error: " <<
        sqrt(amgcl::backend::inner_product(r,r)) /
        sqrt(amgcl::backend::inner_product(f,f)) << std::endl;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    using std::string;
    using std::vector;

    namespace po = boost::program_options;
    namespace io = amgcl::io;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        ("matrix,A", po::value<string>()->required(), "The system matrix")
        ("rhs,f", po::value<string>()->required(),    "The right-hand side")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

    size_t rows;
    vector<ptrdiff_t> ptr, col;
    vector<double> val, rhs;
    std::vector<char> pm;

    {
        auto t = prof.scoped_tic("reading");
        size_t n, m;

        string Afile = vm["matrix"].as<string>();
        string bfile = vm["rhs"].as<string>();

        io::read_crs(Afile, rows, ptr, col, val);
        io::read_dense(bfile, n, m, rhs);
        precondition(n == rows && m == 1, "The RHS vector has wrong size");
    }

    solve(std::tie(rows, ptr, col, val), rhs);

    std::cout << prof << std::endl;
}
