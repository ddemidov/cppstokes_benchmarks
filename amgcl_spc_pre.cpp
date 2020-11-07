#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/make_block_solver.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/solver/idrs.hpp>
#include <amgcl/solver/preonly.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>

#include <amgcl/io/binary.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
template <class Matrix>
void solve_schur(const Matrix &K, const std::vector<double> &rhs, boost::property_tree::ptree &prm, bool quiet)
{
    auto t1 = prof.scoped_tic("schur_complement");

#ifdef PRECOND_SCALAR
    typedef PRECOND_SCALAR prec_scalar;
#else
    typedef float prec_scalar;
#endif

#ifdef BLOCK_U
    typedef amgcl::backend::builtin<amgcl::static_matrix<prec_scalar, 3, 3>> UBackend;
    typedef
        amgcl::make_block_solver<
            amgcl::amg<
                UBackend,
                amgcl::coarsening::aggregation,
                amgcl::relaxation::ilut
                >,
            amgcl::solver::preonly<UBackend>
            > USolver;
#else
    typedef amgcl::backend::builtin<prec_scalar> UBackend;
    typedef
        amgcl::make_solver<
            amgcl::amg<
                UBackend,
                amgcl::coarsening::aggregation,
                amgcl::relaxation::ilut
                >,
            amgcl::solver::preonly<UBackend>
            > USolver;
#endif

    typedef amgcl::backend::builtin<prec_scalar> PBackend;
    typedef amgcl::backend::builtin<double> SBackend;

    prof.tic("setup");
    amgcl::make_solver<
        amgcl::preconditioner::schur_pressure_correction<
            USolver,
            amgcl::make_solver<
                amgcl::relaxation::as_preconditioner<
                    PBackend,
                    amgcl::relaxation::spai0
                    >,
                amgcl::solver::preonly<PBackend>
                >
            >,
        amgcl::solver::idrs<SBackend>
        > solve(K, prm);
    prof.toc("setup");

    if (!quiet) std::cout << solve << std::endl;

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
        ("udofs,u", po::value<int>()->required(), "Number of U DOFs")
        ("params,P", po::value<string>(), "Parameter file in JSON format")
        ("quiet,q", po::bool_switch()->default_value(false), "Suppress solver structure report")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

    boost::property_tree::ptree prm;
    if (vm.count("params"))
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

    auto A = std::tie(rows, ptr, col, val);

    prm.put("solver.s", 5);
    prm.put("solver.maxiter", 1000);
    prm.put("solver.tol", 1e-12);
    prm.put("solver.replacement", true);
    prm.put("solver.smoothing", true);
    prm.put("precond.pmask_size", amgcl::backend::rows(A));
    prm.put("precond.pmask_pattern", ">" + std::to_string(vm["udofs"].as<int>()));
    prm.put("precond.simplec_dia", false);

    solve_schur(A, rhs, prm, vm["quiet"].as<bool>());

    std::cout << prof << std::endl;
}
