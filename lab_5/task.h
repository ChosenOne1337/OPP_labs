#pragma once

#include <cmath>
#include <random>
#include <stdexcept>
#include <cstddef>
#include <mpich/mpi.h>

struct Task {
    Task() = default;

    Task(double leftBoundary, double rightBoundary, unsigned long long dotsNumber):
        leftBoundary{leftBoundary},
        rightBoundary{rightBoundary},
        dotsNumber{dotsNumber} {}

    double operator() () const {
        std::random_device rd;
        std::mt19937 generator{rd()};
        std::uniform_real_distribution<double> distribution{leftBoundary, rightBoundary};

        double funcMean = 0.0;
        for (unsigned long long i = 0; i < dotsNumber; ++i) {
            funcMean += std::log(distribution(generator)) / dotsNumber;
        }

        return (rightBoundary - leftBoundary) * funcMean;
    }

    double leftBoundary = 0.0;
    double rightBoundary = 0.0;
    unsigned long long dotsNumber = 0;
};

class TaskDatatype {
public:
    TaskDatatype(const TaskDatatype&) = delete;
    TaskDatatype(TaskDatatype&&) = delete;
    TaskDatatype& operator=(TaskDatatype) = delete;

    TaskDatatype() {
        const int NITEMS = 3;
        int blockLengths[NITEMS] = {1, 1, 1};
        MPI_Datatype types[NITEMS] = {MPI_DOUBLE, MPI_DOUBLE, MPI_UNSIGNED_LONG_LONG};
        MPI_Datatype &datatype = get_datatype();
        MPI_Aint offsets[NITEMS];

        offsets[0] = offsetof(Task, leftBoundary);
        offsets[1] = offsetof(Task, rightBoundary);
        offsets[2] = offsetof(Task, dotsNumber);

        MPI_Type_create_struct(NITEMS, blockLengths, offsets, types, &datatype);
        MPI_Type_commit(&datatype);
    }

    MPI_Datatype& get_datatype() {
        return datatype;
    }

    ~TaskDatatype() noexcept {
        if (datatype != MPI_DATATYPE_NULL) {
            MPI_Type_free(&datatype);
            datatype = MPI_DATATYPE_NULL;
        }
    }
private:
    MPI_Datatype datatype = MPI_DATATYPE_NULL;
};
