#include <iostream>
#include <libgen.h>
#include <mpich/mpi.h>
#include "task_executor.h"

#define MPI_EXIT(PROC_RANK, EXIT_CODE, ACTION, ...)     \
    {                                                   \
        if ((PROC_RANK) == 0)                           \
        {                                               \
            ACTION(__VA_ARGS__);                        \
        }                                               \
        MPI_Finalize();                                 \
        exit(EXIT_CODE);                                \
    }

namespace Argv {
    enum {
        ProgPathIndex,
        ThreadsNumberIndex,
        LeftBoundaryIndex,
        RightBoundaryIndex,
        DotsNumberIndex,
        RequiredArgc
    };
}

std::size_t parse_parameters(char *argv[], Task &task) {
    std::size_t threadsNumber = 0;
    try {
        threadsNumber = std::stoul(argv[Argv::ThreadsNumberIndex]);
        task.leftBoundary = std::stod(argv[Argv::LeftBoundaryIndex]);
        task.rightBoundary = std::stod(argv[Argv::RightBoundaryIndex]);
        task.dotsNumber = std::stoull(argv[Argv::DotsNumberIndex]);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("Failed to parse parameters");
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("One of the parameters if out of range");
    }

    if (threadsNumber == 0) {
        throw std::invalid_argument("Number of threads has to be > 0");
    }

    if (task.leftBoundary <= 0.0) {
        throw std::invalid_argument("Left boundary has to be > 0.0");
    }

    return threadsNumber;
}

void print_usage(char *progPath) {
    fprintf(stderr, "Usage: %s <threads per process> <left boundary> "
                    "<right boundary> <dots number>\n", basename(progPath));
}

int main(int argc, char *argv[]) {
    int threadLevel;
    int procRank, procNum;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &threadLevel);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    if (argc != Argv::RequiredArgc) {
        MPI_EXIT(procRank, EXIT_FAILURE,
                 print_usage, argv[Argv::ProgPathIndex]);
    }

    if (threadLevel != MPI_THREAD_MULTIPLE) {
        MPI_EXIT(procRank, EXIT_FAILURE,
                 fprintf, stderr, "Error: Can't provide required thread level\n");
    }

    try {
        Task task;
        std::size_t threadsNumber = parse_parameters(argv, task);

        double beg = MPI_Wtime();
        TaskExecutor taskExecutor;
        double result = taskExecutor.execute_task(task, threadsNumber);
        double end = MPI_Wtime();

        if (procRank == 0) {
            printf("Result is %.10f\n"
                   "Elapsed time: %.3fs\n",
                   result, end - beg);
        }
    } catch (const std::invalid_argument &ex) {
        MPI_EXIT(procRank, EXIT_FAILURE,
                 fprintf, stderr, "Error: %s\n", ex.what());
    } catch (const std::runtime_error &ex) {
        MPI_EXIT(procRank, EXIT_FAILURE,
                 fprintf, stderr, "Error: %s\n", ex.what());
    }

    MPI_Finalize();
}
