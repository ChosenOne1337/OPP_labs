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

    int coords[Coords::TotalDims];
    int dims[Coords::TotalDims];
    int gridRank;
    int rootRank = 0;
    MPI_Comm gridComm;

    double stepX, stepY, stepZ;

    double parameterA = 1e+5;

    double phi(double x, double y, double z) {
        return x * x + y * y + z * z;
    }

    double rho(double x, double y, double z) {
        return 6 - parameterA * phi(x, y, z);
    }

    void create_grid_communicator(int procNum) {
        MPI_Dims_create(procNum, Coords::TotalDims, dims);

        int periods[Coords::TotalDims] = {false, false};
        int reorder = true;

        MPI_Cart_create(MPI_COMM_WORLD, Coords::TotalDims, dims, periods, reorder, &gridComm);
        MPI_Cart_get(gridComm, Coords::TotalDims, dims, periods, coords);
        MPI_Cart_rank(gridComm, coords, &gridRank);
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
        const static double reciprocalSquaredStepX = stepX * stepX;
        const static double reciprocalSquaredStepY = stepY * stepY;
        const static double reciprocalSquaredStepZ = stepZ * stepZ;
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

    double getMaxDelta(DiscreteFunc &func1, DiscreteFunc &func2) {
        // grids and domains are supposed to be the same
        const Grid &grid = func1.getGrid();

        double maxDelta = 0.0, delta = 0.0;
        for (int z = 0; z < grid.nodesZ; ++z) {
            for (int y = 0; y < grid.nodesY; ++y) {
                for (int x = 0; x < grid.nodesX; ++x) {
                    delta = std::abs(func1.getValue(x, y, z) - func2.getValue(x, y, z));
                    if (delta > maxDelta) {
                        maxDelta = delta;
                    }
                }
            }
        }

        return maxDelta;
    }

    double check_result(DiscreteFunc &func) {
        const Domain &domain = func.getDomain();
        const Grid &grid = func.getGrid();

        DiscreteFunc realFunc(domain, grid);
        realFunc.setValues(phi);

        return getMaxDelta(realFunc, func);
    }

} // anonymous namespace

void calculate(int procNum, int nodesX, int nodesY, int nodesZ, double eps) {
    create_grid_communicator(procNum);

    double x0 = -1.0, y0 = -1.0, z0 = -1.0;
    double lenX = 2.0, lenY = 2.0, lenZ = 2.0;
    Domain domain(x0, y0, z0, lenX, lenY, lenZ);
    Grid grid(nodesX, nodesY, nodesZ);
    DiscreteFunc fullFunc = (gridRank == rootRank) ? DiscreteFunc(domain, grid) : DiscreteFunc();

    stepX = domain.extent.lenX / (grid.nodesX - 1.0);
    stepY = domain.extent.lenY / (grid.nodesY - 1.0);
    stepZ = domain.extent.lenZ / (grid.nodesZ - 1.0);

    DiscreteFunc currPartialFunc = getPartialFunc(domain, grid);
    DiscreteFunc prevPartialFunc = getPartialFunc(domain, grid);
    setEdgeValues(currPartialFunc);

    struct {
        double val;
        int rank;
    } maxDelta, localDelta;
    localDelta.rank = gridRank;

    double beg = MPI_Wtime();

    do {
        std::swap(currPartialFunc, prevPartialFunc);

        // send shadow edges

        calculateNextIteration(currPartialFunc, prevPartialFunc);

        // wait until processes exchange shadow edges

        calculateRemainder(currPartialFunc, prevPartialFunc);

        // calculate delta
        localDelta.val = getMaxDelta(currPartialFunc, prevPartialFunc);
        MPI_Allreduce(&localDelta, &maxDelta, 1, MPI_DOUBLE_INT, MPI_MAXLOC, gridComm);
    } while (maxDelta.val >= eps);

    double end = MPI_Wtime();

    if (gridRank == rootRank) {
        printf("Elapsed time: %.3f\n", end - beg);
        double error = check_result(fullFunc);
        printf("Maximum delta: %g\n", error);
    }

    MPI_Comm_free(&gridComm);
}
