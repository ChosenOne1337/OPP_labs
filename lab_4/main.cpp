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
              << " <N1> <N2> <N3> <eps>" << std::endl;
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

void check_parameters(int procNum, int N1, int N2, int N3, double eps) {
    if (N1 < procNum) {
        throw std::invalid_argument("N1 is less than number of processes");
    }
    if (N2 < procNum) {
        throw std::invalid_argument("N2 is less than number of processes");
    }
    if (N3 < procNum) {
        throw std::invalid_argument("N3 is less than number of processes");
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
        int N1, N2, N3;
        parse_parameters(argv, N1, N2, N3, eps);

        int procNum = 0;
        MPI_Init(nullptr, nullptr);
        MPI_Comm_size(MPI_COMM_WORLD, &procNum);

        check_parameters(procNum, N1, N2, N3, eps);
        calculate(procNum, N1, N2, N3, eps);

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
