#include <mpich/mpi.h>
#include <memory>
#include <cmath>
#include "calculations.h"
#include "discretefunc.h"
#include "utils.h"

namespace {
    namespace Coords {
        enum {
            RowDim,
            ColDim,
            TotalDims
        };
    }

    namespace Request {
        enum {
            LeftIndex,
            TopIndex,
            RightIndex,
            BottomIndex,
            TotalRequests
        };
    }

    namespace Neighbor {
        enum {
            LeftIndex,
            TopIndex,
            RightIndex,
            BottomIndex,
            TotalNeighbors
        };
    }

    namespace Edge {
        enum {
            LeftIndex,
            TopIndex,
            RightIndex,
            BottomIndex,
            TotalEdges
        };
    }



    MPI_Comm gridComm;
    int coords[Coords::TotalDims];
    int dims[Coords::TotalDims];
    int gridRank;
    int rootRank = 0;
    int neighborsRanks[Neighbor::TotalNeighbors];

    MPI_Request sendRequests[Request::TotalRequests] = {
        MPI_REQUEST_NULL, MPI_REQUEST_NULL,
        MPI_REQUEST_NULL, MPI_REQUEST_NULL
    };
    MPI_Request recvRequests[Request::TotalRequests] = {
        MPI_REQUEST_NULL, MPI_REQUEST_NULL,
        MPI_REQUEST_NULL, MPI_REQUEST_NULL
    };

    MPI_Datatype edgesTypes[Edge::TotalEdges];

    double stepX, stepY, stepZ;
    double parameterA = 1e+5;



    double phi(double x, double y, double z) {
        return x * x + y * y + z * z;
    }

    double rho(double x, double y, double z) {
        return 6.0 - parameterA * phi(x, y, z);
    }



    void create_grid_communicator(int procNum) {
        MPI_Dims_create(procNum, Coords::TotalDims, dims);

        int periods[Coords::TotalDims] = {false, false};
        int reorder = true;

        MPI_Cart_create(MPI_COMM_WORLD, Coords::TotalDims, dims, periods, reorder, &gridComm);
        MPI_Cart_get(gridComm, Coords::TotalDims, dims, periods, coords);
        MPI_Cart_rank(gridComm, coords, &gridRank);

        MPI_Cart_shift(gridComm, Coords::RowDim, 1,
                       &neighborsRanks[Neighbor::LeftIndex], &neighborsRanks[Neighbor::RightIndex]);
        MPI_Cart_shift(gridComm, Coords::ColDim, 1,
                       &neighborsRanks[Neighbor::TopIndex], &neighborsRanks[Neighbor::BottomIndex]);
    }

    void free_grid_communicator() {
        MPI_Comm_free(&gridComm);
    }



    void create_edge_datatypes(const Grid &grid) {
        for (MPI_Datatype &type: edgesTypes) {
            type = MPI_DATATYPE_NULL;
        }

        int blocksCount = 1;
        int blockLength = grid.nodesX * grid.nodesY;
        int stride = 0;

        if (coords[Coords::RowDim] != 0) {
            MPI_Type_vector(blocksCount, blockLength, stride, MPI_DOUBLE, &edgesTypes[Edge::TopIndex]);
        }

        if (coords[Coords::RowDim] != dims[Coords::RowDim] - 1) {
            MPI_Type_vector(blocksCount, blockLength, stride, MPI_DOUBLE, &edgesTypes[Edge::BottomIndex]);
        }


        blocksCount = grid.nodesZ;
        blockLength = grid.nodesX;
        stride = grid.nodesX * grid.nodesY;

        MPI_Datatype newType;
        MPI_Aint lb;
        MPI_Aint doubleExtent;
        MPI_Type_extent(MPI_DOUBLE, &doubleExtent);

        if (coords[Coords::ColDim] != 0) {
            MPI_Type_vector(blocksCount, blockLength, stride, MPI_DOUBLE, &newType);
            MPI_Type_lb(newType, &lb);
            MPI_Type_create_resized(newType, lb, doubleExtent * blockLength, &edgesTypes[Edge::LeftIndex]);
        }

        if (coords[Coords::ColDim] != dims[Coords::ColDim] - 1) {
            MPI_Type_vector(blocksCount, blockLength, stride, MPI_DOUBLE, &newType);
            MPI_Type_lb(newType, &lb);
            MPI_Type_create_resized(newType, lb, doubleExtent * blockLength, &edgesTypes[Edge::RightIndex]);
        }


        for (MPI_Datatype &type: edgesTypes) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_commit(&type);
            }
        }
    }

    void free_edge_datatypes() {
        for (MPI_Datatype &type: edgesTypes) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_free(&type);
            }
        }
    }



    void addShadowEdges(Domain &domain, Grid &grid) {
        if (coords[Coords::RowDim] != 0) {
            domain.extent.lenZ += stepZ;
            grid.nodesZ += 1;
        }

        if (coords[Coords::RowDim] != dims[Coords::RowDim] - 1) {
            domain.extent.lenZ += stepZ;
            domain.origin.z -= stepZ;
            grid.nodesZ += 1;
        }

        if (coords[Coords::ColDim] != 0) {
            domain.extent.lenY += stepY;
            domain.origin.y -= stepY;
            grid.nodesY += 1;
        }

        if (coords[Coords::ColDim] != dims[Coords::ColDim] - 1) {
            domain.extent.lenY += stepY;
            grid.nodesY += 1;
        }
    }

    DiscreteFunc getPartialFunc(const Domain &domain, const Grid &grid) {
        int newNodesX = grid.nodesX;
        int newNodesY = get_chunk_size(coords[Coords::ColDim], dims[Coords::ColDim], grid.nodesY);
        int newNodesZ = get_chunk_size(dims[Coords::RowDim] - coords[Coords::RowDim] - 1,
                                            dims[Coords::RowDim], grid.nodesZ);

        int nodesOffsetX = 0;
        int nodesOffsetY = get_chunk_offset(coords[Coords::ColDim], dims[Coords::ColDim], grid.nodesY);
        int nodesOffsetZ = get_chunk_offset(dims[Coords::RowDim] - coords[Coords::RowDim] - 1,
                                                dims[Coords::RowDim], grid.nodesZ);

        double newX0 = domain.origin.x + nodesOffsetX * stepX;
        double newY0 = domain.origin.y + nodesOffsetY * stepY;
        double newZ0 = domain.origin.z + nodesOffsetZ * stepZ;
        double newLenX = (newNodesX - 1) * stepX;
        double newLenY = (newNodesY - 1) * stepY;
        double newLenZ = (newNodesZ - 1) * stepZ;

        Domain partialDomain(newX0, newY0, newZ0, newLenX, newLenY, newLenZ);
        Grid partialGrid(newNodesX, newNodesY, newNodesZ);

        addShadowEdges(partialDomain, partialGrid);

        return DiscreteFunc(partialDomain, partialGrid);
    }

    void setEdgeValues(DiscreteFunc &partialFunc) {
        partialFunc.setFrontEdgeValues(phi);
        partialFunc.setBackEdgeValues(phi);

        if (coords[Coords::RowDim] == 0) {
            partialFunc.setTopEdgeValues(phi);
        }

        if (coords[Coords::RowDim] == dims[Coords::RowDim] - 1) {
            partialFunc.setBottomEdgeValues(phi);
        }

        if (coords[Coords::ColDim] == 0) {
            partialFunc.setLeftEdgeValues(phi);
        }

        if (coords[Coords::ColDim] == dims[Coords::ColDim] - 1) {
            partialFunc.setRightEdgeValues(phi);
        }
    }



    double iterationFunction(DiscreteFunc &prevFunc, int x, int y, int z) {
        const static double reciprocalSquaredStepX = 1.0 / (stepX * stepX);
        const static double reciprocalSquaredStepY = 1.0 / (stepY * stepY);
        const static double reciprocalSquaredStepZ = 1.0 / (stepZ * stepZ);
        const static double multiplier = 1.0 / (2.0 * reciprocalSquaredStepX + 2.0 * reciprocalSquaredStepY
                                                + 2.0 * reciprocalSquaredStepZ + parameterA);

        double partX = reciprocalSquaredStepX * (prevFunc.getValue(x + 1, y, z) + prevFunc.getValue(x - 1, y, z));
        double partY = reciprocalSquaredStepY * (prevFunc.getValue(x, y + 1, z) + prevFunc.getValue(x, y - 1, z));
        double partZ = reciprocalSquaredStepZ * (prevFunc.getValue(x, y, z + 1) + prevFunc.getValue(x, y, z - 1));

        return multiplier * (partX + partY + partZ - rho(x, y, z));
    }

    void calculateNextIteration(DiscreteFunc &currPartialFunc, DiscreteFunc &prevPartialFunc) {
        const Grid &grid = currPartialFunc.getGrid();
        if (grid.nodesX == 2 || grid.nodesY == 2 || grid.nodesZ == 2) {
            // grid consists of border and shadow edges
            return;
        }

        double val = 0.0;
        for (int z = 1; z < grid.nodesZ - 1; ++z) {
            for (int y = 1; y < grid.nodesY - 1; ++y) {
                for (int x = 1; x < grid.nodesX - 1; ++x) {
                    val = iterationFunction(prevPartialFunc, x, y, z);
                    currPartialFunc.setValue(val, x, y, z);
                }
            }
        }
    }

    void calculateRemainder(DiscreteFunc &currPartialFunc, DiscreteFunc &prevPartialFunc) {
        const Grid &grid = currPartialFunc.getGrid();
        if (grid.nodesX == 2 || grid.nodesY == 2 || grid.nodesZ == 2) {
            // grid consists of border and shadow edges
            return;
        }

        double val = 0.0;

        if (coords[Coords::RowDim] != 0) {
            int z = grid.nodesZ - 2;
            for (int y = 1; y < grid.nodesY - 1; ++y) {
                for (int x = 1; x < grid.nodesX - 1; ++x) {
                    val = iterationFunction(prevPartialFunc, x, y, z);
                    currPartialFunc.setValue(val, x, y, z);
                }
            }
        }

        if (coords[Coords::RowDim] != dims[Coords::RowDim] - 1) {
            int z = 1;
            for (int y = 1; y < grid.nodesY - 1; ++y) {
                for (int x = 1; x < grid.nodesX - 1; ++x) {
                    val = iterationFunction(prevPartialFunc, x, y, z);
                    currPartialFunc.setValue(val, x, y, z);
                }
            }
        }

        if (coords[Coords::ColDim] != 0) {
            int y = 1;
            for (int z = 1; z < grid.nodesZ - 1; ++z) {
                for (int x = 1; x < grid.nodesX - 1; ++x) {
                    val = iterationFunction(prevPartialFunc, x, y, z);
                    currPartialFunc.setValue(val, x, y, z);
                }
            }
        }

        if (coords[Coords::ColDim] != dims[Coords::ColDim] - 1) {
            int y = grid.nodesY - 2;
            for (int z = 1; z < grid.nodesZ - 1; ++z) {
                for (int x = 1; x < grid.nodesX - 1; ++x) {
                    val = iterationFunction(prevPartialFunc, x, y, z);
                    currPartialFunc.setValue(val, x, y, z);
                }
            }
        }
    }



    void exchange_edges(DiscreteFunc &partialFunc, int edgeIndex, int neighborIndex, int requestIndex) {
        const static int tag = 42;
        const static int sendCount = 1, recvCount = 1;

        const Grid &grid = partialFunc.getGrid();
        int sendDisplacement = 0, recvDisplacement = 0;

        switch (edgeIndex) {
        case Edge::LeftIndex:
            recvDisplacement = 0;
            sendDisplacement = 1;
            break;
        case Edge::RightIndex:
            recvDisplacement = grid.nodesY - 1;
            sendDisplacement = grid.nodesY - 2;
            break;
        case Edge::TopIndex:
            recvDisplacement = grid.nodesZ - 1;
            sendDisplacement = grid.nodesZ - 2;
            break;
        case Edge::BottomIndex:
            recvDisplacement = 0;
            sendDisplacement = 1;
            break;
        default:
            return;
        }

        MPI_Isend(partialFunc.getData() + sendDisplacement, sendCount, edgesTypes[edgeIndex],
                neighborsRanks[neighborIndex], tag, gridComm, &sendRequests[requestIndex]);
        MPI_Irecv(partialFunc.getData() + recvDisplacement, recvCount, edgesTypes[edgeIndex],
                neighborsRanks[neighborIndex], tag, gridComm, &recvRequests[requestIndex]);
    }

    void initiate_edge_interchange(DiscreteFunc &partialFunc) {
        if (coords[Coords::RowDim] != 0) {
            exchange_edges(partialFunc, Edge::TopIndex, Neighbor::TopIndex, Request::TopIndex);
        }

        if (coords[Coords::RowDim] != dims[Coords::RowDim] - 1) {
            exchange_edges(partialFunc, Edge::BottomIndex, Neighbor::BottomIndex, Request::BottomIndex);
        }

        if (coords[Coords::ColDim] != 0) {
            exchange_edges(partialFunc, Edge::LeftIndex, Neighbor::LeftIndex, Request::LeftIndex);
        }

        if (coords[Coords::ColDim] != dims[Coords::ColDim] - 1) {
            exchange_edges(partialFunc, Edge::RightIndex, Neighbor::RightIndex, Request::RightIndex);
        }
    }

    void finish_edge_interchange() {
        MPI_Waitall(Neighbor::TotalNeighbors, recvRequests, MPI_STATUSES_IGNORE);
    }



    double getLocalDiff(DiscreteFunc &func1, DiscreteFunc &func2) {
        // grids and domains are supposed to be the same
        const Grid &grid = func1.getGrid();
        if (grid.nodesX == 2 || grid.nodesY == 2 || grid.nodesZ == 2) {
            // grid consists of border and shadow edges
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

    double getMaxDiff(DiscreteFunc &func1, DiscreteFunc &func2) {
        struct {
            double val;
            int rank;
        } maxDiff, localDiff;

        localDiff.rank = gridRank;
        localDiff.val = getLocalDiff(func1, func2);
        MPI_Allreduce(&localDiff, &maxDiff, 1, MPI_DOUBLE_INT, MPI_MAXLOC, gridComm);

        return maxDiff.val;
    }

    double check_result(DiscreteFunc &partialFunc) {
        const Domain &domain = partialFunc.getDomain();
        const Grid &grid = partialFunc.getGrid();

        DiscreteFunc truePartialFunc(domain, grid);
        truePartialFunc.setValues(phi);

        return getLocalDiff(truePartialFunc, partialFunc);
    }

} // anonymous namespace



void calculate(int procNum, int nodesX, int nodesY, int nodesZ, double eps) {
    double x0 = -1.0, y0 = -1.0, z0 = -1.0;
    double lenX = 2.0, lenY = 2.0, lenZ = 2.0;
    Domain domain(x0, y0, z0, lenX, lenY, lenZ);
    Grid grid(nodesX, nodesY, nodesZ);

    stepX = domain.extent.lenX / (grid.nodesX - 1.0);
    stepY = domain.extent.lenY / (grid.nodesY - 1.0);
    stepZ = domain.extent.lenZ / (grid.nodesZ - 1.0);

    create_grid_communicator(procNum);

    DiscreteFunc currPartialFunc = getPartialFunc(domain, grid);
    DiscreteFunc prevPartialFunc = getPartialFunc(domain, grid);
    setEdgeValues(currPartialFunc);

    create_edge_datatypes(currPartialFunc.getGrid());

    double maxDiff = 0.0;
    double beg = MPI_Wtime();

    do {
        std::swap(currPartialFunc, prevPartialFunc);

        initiate_edge_interchange(prevPartialFunc);
        calculateNextIteration(currPartialFunc, prevPartialFunc);
        finish_edge_interchange();

        calculateRemainder(currPartialFunc, prevPartialFunc);
        maxDiff = getMaxDiff(currPartialFunc, prevPartialFunc);
    } while (maxDiff >= eps);

    double end = MPI_Wtime();

    double delta = check_result(currPartialFunc);
    if (gridRank == rootRank) {
        printf("Elapsed time: %.3f\n", end - beg);
        printf("Delta: %g\n", delta);
    }

    free_edge_datatypes();
    free_grid_communicator();
}
