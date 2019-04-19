#include <iostream>
#include <stdexcept>
#include <exception>
#include <libgen.h>
#include "calculations.h"
#include <mpich/mpi.h>

namespace Argv {
    enum {
        ProgPathIndex,
        N1_Index, N2_Index, N3_Index,
        Eps_Index,
        RequiredArgc
    };
}

void print_usage(char *progPath) {
    std::cerr << "Usage: " << basename(progPath)
              << " <Nx> <Ny> <Nz> <eps>" << std::endl;
}

void parse_parameters(char *argv[], int &N1, int &N2, int &N3, double &eps) {
    try {
        N1 = std::stoi(argv[Argv::N1_Index]);
        N2 = std::stoi(argv[Argv::N2_Index]);
        N3 = std::stoi(argv[Argv::N3_Index]);
        eps = std::stod(argv[Argv::Eps_Index]);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("failed to parse one of the arguments");
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("one of the arguments is out of range");
    }
}

void check_parameters(int procNum, int Nx, int Ny, int Nz, double eps) {
    if (Nx <= 1) {
        throw std::invalid_argument("(Nx - 1) is less than or equal to 0");
    }
    if (Ny - 1 < procNum) {
        throw std::invalid_argument("(Ny - 1) is less than number of processes");
    }
    if (Nz - 1 < procNum) {
        throw std::invalid_argument("(Nz - 1) is less than number of processes");
    }
    if (Nx == 2 || Ny == 2 || Nz == 2) {
        throw std::invalid_argument("One of the grid dimension's size == 2, "
                                    "i.e. grid contains only its border");
    }
    if (eps <= 0.0) {
        throw std::invalid_argument("eps is <= 0.0");
    }
}

int main(int argc, char *argv[]) {
    if (argc != Argv::RequiredArgc) {
        char *progPath = argv[Argv::ProgPathIndex];
        print_usage(progPath);
        return EXIT_FAILURE;
    }

    try {
        double eps;
        int Nx, Ny, Nz;
        parse_parameters(argv, Nx, Ny, Nz, eps);

        int procNum = 0;
        MPI_Init(nullptr, nullptr);
        MPI_Comm_size(MPI_COMM_WORLD, &procNum);

        check_parameters(procNum, Nx, Ny, Nz, eps);
        calculate(procNum, Nx, Ny, Nz, eps);

        MPI_Finalize();
    } catch (const std::invalid_argument &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::runtime_error &ex) {
       std::cerr << "Error: " << ex.what() << std::endl;
       return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
