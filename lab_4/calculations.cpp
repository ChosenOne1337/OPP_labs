#include <mpich/mpi.h>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include "calculations.h"
#include "discretefunc.h"
#include "utils.h"

namespace  {
    int procNum;
    int procRank;
    int rootRank = 0;
}

void init_values();
DiscreteFunc get_partial_func(int, int, int);
void initiate_edge_interchange(DiscreteFunc&);
void calculate_next_iteration(DiscreteFunc&, DiscreteFunc&);
void finish_edge_interchange();
void calculate_remainder(DiscreteFunc&, DiscreteFunc&);
double get_max_diff(DiscreteFunc&, DiscreteFunc&);
double check_result(DiscreteFunc&);

void calculate(int nodesX, int nodesY, int nodesZ, double eps) {
    init_values();

    DiscreteFunc currPartialFunc = get_partial_func(nodesX, nodesY, nodesZ);
    DiscreteFunc prevPartialFunc = currPartialFunc;

    double maxDiff = 0.0;
    double beg = MPI_Wtime();

    do {
        swap(currPartialFunc, prevPartialFunc);

        initiate_edge_interchange(prevPartialFunc);
        calculate_next_iteration(currPartialFunc, prevPartialFunc);
        finish_edge_interchange();

        calculate_remainder(currPartialFunc, prevPartialFunc);
        maxDiff = get_max_diff(currPartialFunc, prevPartialFunc);
    } while (maxDiff >= eps);

    double end = MPI_Wtime();

    double delta = check_result(currPartialFunc);
    if (procRank == rootRank) {
        printf("Elapsed time: %.3f\n", end - beg);
        printf("Delta: %g\n", delta);
    }
}



namespace Request {
    enum {
        TopIndex,
        BottomIndex,
        TotalRequests
    };
}

namespace Neighbor {
    enum {
        TopIndex,
        BottomIndex,
        TotalNeighbors
    };
}

namespace  {
    int neighborsRanks[Neighbor::TotalNeighbors];

    MPI_Request sendRequests[Request::TotalRequests] = {
        MPI_REQUEST_NULL, MPI_REQUEST_NULL,
    };
    MPI_Request recvRequests[Request::TotalRequests] = {
        MPI_REQUEST_NULL, MPI_REQUEST_NULL,
    };

    double stepX, stepY, stepZ;
    double parameterA = 1e+5;
}

void add_shadow_edges(Domain&, Grid&);
void set_edge_values(DiscreteFunc&);
void exchange_edges(DiscreteFunc&, int, int);
double iteration_function(DiscreteFunc&, int, int, int);
void calculate_slice(DiscreteFunc&, DiscreteFunc&, int);



inline double phi(double x, double y, double z) {
    return x * x + y * y + z * z;
}

inline double rho(double x, double y, double z) {
    return 6.0 - parameterA * phi(x, y, z);
}



void init_values() {
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    neighborsRanks[Neighbor::TopIndex] = procRank + 1;
    neighborsRanks[Neighbor::BottomIndex] = procRank - 1;
}

DiscreteFunc get_partial_func(int nodesX, int nodesY, int nodesZ) {
    const static double x0 = -1.0, y0 = -1.0, z0 = -1.0;
    const static double lenX = 2.0, lenY = 2.0, lenZ = 2.0;

    stepX = lenX / (nodesX - 1);
    stepY = lenY / (nodesY - 1);
    stepZ = lenZ / (nodesZ - 1);

    int newNodesZ = get_chunk_size(procRank, procNum, nodesZ);
    int nodesOffsetZ = get_chunk_offset(procRank, procNum, nodesZ);
    double newZ0 = z0 + nodesOffsetZ * stepZ;
    double newLenZ = (newNodesZ - 1) * stepZ;

    Domain partialDomain(x0, y0, newZ0, lenX, lenY, newLenZ);
    Grid partialGrid(nodesX, nodesY, newNodesZ);

    add_shadow_edges(partialDomain, partialGrid);
    DiscreteFunc partialFunc(partialDomain, partialGrid);
    set_edge_values(partialFunc);

    return partialFunc;
}

void add_shadow_edges(Domain &domain, Grid &grid) {
    if (procRank != rootRank) {
        domain.origin.z -= stepZ;
        domain.extent.lenZ += stepZ;
        grid.nodesZ += 1;
    }

    if (procRank != procNum - 1) {
        domain.extent.lenZ += stepZ;
        grid.nodesZ += 1;
    }
}

void set_edge_values(DiscreteFunc &partialFunc) {
    partialFunc.setFrontEdgeValues(phi);
    partialFunc.setBackEdgeValues(phi);
    partialFunc.setLeftEdgeValues(phi);
    partialFunc.setRightEdgeValues(phi);

    if (procRank == rootRank) {
        partialFunc.setBottomEdgeValues(phi);
    }

    if (procRank == procNum - 1) {
        partialFunc.setTopEdgeValues(phi);
    }
}



void initiate_edge_interchange(DiscreteFunc &partialFunc) {
    if (procRank != rootRank) {
        exchange_edges(partialFunc, Neighbor::BottomIndex, Request::BottomIndex);
    }

    if (procRank != procNum - 1) {
        exchange_edges(partialFunc, Neighbor::TopIndex, Request::TopIndex);
    }
}

void exchange_edges(DiscreteFunc &partialFunc, int neighborIndex, int requestIndex) {
    const static int tag = 42;

    const Grid &grid = partialFunc.getGrid();
    int sendDisplacement = 0, recvDisplacement = 0;
    int sendCount = grid.nodesX * grid.nodesY;
    int recvCount = sendCount;

    switch (neighborIndex) {
    case Neighbor::TopIndex:
        recvDisplacement = grid.nodesX * grid.nodesY * (grid.nodesZ - 1);
        sendDisplacement = grid.nodesX * grid.nodesY * (grid.nodesZ - 2);
        break;
    case Neighbor::BottomIndex:
        recvDisplacement = 0;
        sendDisplacement = grid.nodesX * grid.nodesY;
        break;
    default:
        return;
    }

    MPI_Isend(partialFunc.getData() + sendDisplacement, sendCount, MPI_DOUBLE,
            neighborsRanks[neighborIndex], tag, MPI_COMM_WORLD, &sendRequests[requestIndex]);
    MPI_Irecv(partialFunc.getData() + recvDisplacement, recvCount, MPI_DOUBLE,
            neighborsRanks[neighborIndex], tag, MPI_COMM_WORLD, &recvRequests[requestIndex]);
}

void finish_edge_interchange() {
    MPI_Waitall(Neighbor::TotalNeighbors, recvRequests, MPI_STATUSES_IGNORE);
}



void calculate_next_iteration(DiscreteFunc &currPartialFunc, DiscreteFunc &prevPartialFunc) {
    const Grid &grid = currPartialFunc.getGrid();
    if (grid.nodesX == 2 || grid.nodesY == 2 || grid.nodesZ == 2) {
        // grid consists of border and shadow edges
        return;
    }

    for (int z = 1; z < grid.nodesZ - 1; ++z) {
        for (int y = 1; y < grid.nodesY - 1; ++y) {
            for (int x = 1; x < grid.nodesX - 1; ++x) {
                double value = iteration_function(prevPartialFunc, x, y, z);
                currPartialFunc.setValue(value, x, y, z);
            }
        }
    }
}

double iteration_function(DiscreteFunc &prevFunc, int x, int y, int z) {
    const static double reciprocalSquaredStepX = 1.0 / (stepX * stepX);
    const static double reciprocalSquaredStepY = 1.0 / (stepY * stepY);
    const static double reciprocalSquaredStepZ = 1.0 / (stepZ * stepZ);
    const static double multiplier = 1.0 / (2.0 * reciprocalSquaredStepX +
                                            2.0 * reciprocalSquaredStepY +
                                            2.0 * reciprocalSquaredStepZ +
                                            parameterA);

    double partX = reciprocalSquaredStepX * (prevFunc.getValue(x + 1, y, z) + prevFunc.getValue(x - 1, y, z));
    double partY = reciprocalSquaredStepY * (prevFunc.getValue(x, y + 1, z) + prevFunc.getValue(x, y - 1, z));
    double partZ = reciprocalSquaredStepZ * (prevFunc.getValue(x, y, z + 1) + prevFunc.getValue(x, y, z - 1));

    const Domain &domain = prevFunc.getDomain();
    double realX = domain.origin.x + stepX * x;
    double realY = domain.origin.y + stepY * y;
    double realZ = domain.origin.z + stepZ * z;

    return multiplier * (partX + partY + partZ - rho(realX, realY, realZ));
}



void calculate_remainder(DiscreteFunc &currPartialFunc, DiscreteFunc &prevPartialFunc) {
    const Grid &grid = currPartialFunc.getGrid();
    if (grid.nodesX == 2 || grid.nodesY == 2 || grid.nodesZ == 2) {
        return;
    }

    if (procRank != rootRank) {
        int levelZ = 1;
        calculate_slice(currPartialFunc, prevPartialFunc, levelZ);
    }

    if (procRank != procNum - 1) {
        int levelZ = grid.nodesZ - 2;
        calculate_slice(currPartialFunc, prevPartialFunc, levelZ);
    }
}

void calculate_slice(DiscreteFunc &currPartialFunc, DiscreteFunc &prevPartialFunc, int levelZ) {
    const Grid &grid = currPartialFunc.getGrid();
    for (int y = 1; y < grid.nodesY - 1; ++y) {
        for (int x = 1; x < grid.nodesX - 1; ++x) {
            double value = iteration_function(prevPartialFunc, x, y, levelZ);
            currPartialFunc.setValue(value, x, y, levelZ);
        }
    }
}



double get_local_diff(DiscreteFunc &func1, DiscreteFunc &func2) {
    // grids and domains are supposed to be the same
    const Grid &grid = func1.getGrid();
    if (grid.nodesX == 2 || grid.nodesY == 2 || grid.nodesZ == 2) {
        // grid consists of a border and shadow edges
        return 0.0;
    }

    double maxLocalDiff = 0.0, localDiff = 0.0;
    for (int z = 1; z < grid.nodesZ - 1; ++z) {
        for (int y = 1; y < grid.nodesY - 1; ++y) {
            for (int x = 1; x < grid.nodesX - 1; ++x) {
                localDiff = std::abs(func1.getValue(x, y, z) - func2.getValue(x, y, z));
                if (localDiff > maxLocalDiff) {
                    maxLocalDiff = localDiff;
                }
            }
        }
    }

    return maxLocalDiff;
}

double get_max_diff(DiscreteFunc &func1, DiscreteFunc &func2) {
    struct {
        double val;
        int rank;
    } maxDiff, localDiff;

    localDiff.rank = procRank;
    localDiff.val = get_local_diff(func1, func2);
    MPI_Allreduce(&localDiff, &maxDiff, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    return maxDiff.val;
}

double check_result(DiscreteFunc &partialFunc) {
    const Domain &domain = partialFunc.getDomain();
    const Grid &grid = partialFunc.getGrid();

    DiscreteFunc truePartialFunc(domain, grid);
    truePartialFunc.setValues(phi);

    return get_max_diff(truePartialFunc, partialFunc);
}
