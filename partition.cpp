#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cassert>

#include <boost/program_options.hpp>

#include <amgcl/util.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>

extern "C" {
#include <metis.h>
}

using amgcl::precondition;

//---------------------------------------------------------------------------
std::vector<int> partition(
        int npart,
        const std::vector<int> &ptr,
        const std::vector<int> &col
        )
{
    int nrows = ptr.size() - 1;

    std::vector<int> part(nrows);

    if (npart == 1) {
        std::fill(part.begin(), part.end(), 0);
    } else {
        int edgecut;

#if defined(METIS_VER_MAJOR) && (METIS_VER_MAJOR >= 5)
        int nconstraints = 1;
        METIS_PartGraphKway(
                &nrows, //nvtxs
                &nconstraints, //ncon -- new
                const_cast<int*>(ptr.data()), //xadj
                const_cast<int*>(col.data()), //adjncy
                NULL, //vwgt
                NULL, //vsize -- new
                NULL, //adjwgt
                &npart,
                NULL,//real t *tpwgts,
                NULL,// real t ubvec
                NULL,
                &edgecut,
                part.data()
                );
#else
        int wgtflag = 0;
        int numflag = 0;
        int options = 0;

        METIS_PartGraphKway(
                &nrows,
                const_cast<int*>(ptr.data()),
                const_cast<int*>(col.data()),
                NULL,
                NULL,
                &wgtflag,
                &numflag,
                &npart,
                &options,
                &edgecut,
                part.data()
                );
#endif
    }

    return part;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;

    try {
        std::string ifile;
        std::string ofile = "partition.mtx";

        int nparts, block_size;

        po::options_description desc("Options");

        desc.add_options()
            ("help,h", "show help")
            ("input,i",      po::value<std::string>(&ifile)->required(), "Input matrix")
            ("output,o",     po::value<std::string>(&ofile)->default_value(ofile), "Output file")
            (
             "binary,B",
             po::bool_switch()->default_value(false),
             "When specified, treat input files as binary instead of as MatrixMarket. "
            )
            ("nparts,n",     po::value<int>(&nparts)->required(), "Number of parts")
            ("block_size,b", po::value<int>(&block_size)->default_value(1), "Block size")
            ;

        po::positional_options_description pd;
        pd.add("input", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pd).run(), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);

        size_t rows;
        std::vector<int> ptr, col;

        bool binary = vm["binary"].as<bool>();

        if (binary) {
            std::ifstream f(ifile, std::ios::binary);
            precondition(f.read((char*)&rows, sizeof(rows)), "Wrong file format?");
            ptr.resize(rows + 1);
            for (size_t i = 0; i <= rows; ++i) {
                ptrdiff_t p;
                precondition(f.read((char*)&p, sizeof(p)), "Wrong file format?");
                ptr[i] = p;
            }
            col.resize(ptr.back());
            for (ptrdiff_t i = 0; i < ptr.back(); ++i) {
                ptrdiff_t p;
                precondition(f.read((char*)&p, sizeof(p)), "Wrong file format?");
                col[i] = p;
            }
        } else {
            std::vector<double> val;
            size_t cols;
            std::tie(rows, cols) = amgcl::io::mm_reader(ifile)(ptr, col, val);
            precondition(rows == cols, "Non-square system matrix");
        }

        std::vector<int> part = partition(nparts, ptr, col);

        if (binary) {
            std::ofstream p(ofile.c_str(), std::ios::binary);

            amgcl::io::write(p, rows);
            amgcl::io::write(p, size_t(1));
            amgcl::io::write(p, part);
        } else {
            amgcl::io::mm_write(ofile, &part[0], part.size());
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
