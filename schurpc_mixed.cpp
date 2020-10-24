#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/make_block_solver.hpp>

#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
template <class Matrix>
void solve_schur(const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm)
{
    auto t1 = prof.scoped_tic("schur_complement");

#ifdef PRECOND_SCALAR
    typedef PRECOND_SCALAR prec_scalar;
#else
    typedef float prec_scalar;
#endif

#ifdef BLOCK_U
    typedef amgcl::backend::builtin<amgcl::static_matrix<prec_scalar, 3, 3>> UBackend;
#else
    typedef amgcl::backend::builtin<prec_scalar> UBackend;
#endif

    typedef amgcl::backend::builtin<prec_scalar> PBackend;
    typedef amgcl::backend::builtin<double> SBackend;

    prof.tic("setup");
    amgcl::make_solver<
        amgcl::preconditioner::schur_pressure_correction<
            amgcl::make_solver<
                amgcl::runtime::preconditioner<UBackend>,
                amgcl::runtime::solver::wrapper<UBackend>
                >,
            amgcl::make_solver<
                amgcl::runtime::preconditioner<PBackend>,
                amgcl::runtime::solver::wrapper<PBackend>
                >
            >,
        amgcl::runtime::solver::wrapper<SBackend>
        > solve(K, prm, bprm);
    prof.toc("setup");

    std::cout << solve << std::endl;

    auto n = amgcl::backend::rows(K);

    amgcl::numa_vector<double> f(rhs);
    amgcl::numa_vector<double> x(n);

    size_t iters;
    double error;

    prof.tic("solve");
    std::tie(iters, error) = solve(K, f, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl;
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
        ("params,P", po::value<string>()->required(), "parameter file in json format")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

    boost::property_tree::ptree prm;
    read_json(vm["params"].as<string>(), prm);

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

    solve_schur(std::tie(rows, ptr, col, val), rhs, prm);

    std::cout << prof << std::endl;
}