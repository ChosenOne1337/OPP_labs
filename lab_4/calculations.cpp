#include <mpich/mpi.h>
#include <memory>
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

    double phi(double x, double y, double z) {
        return x * x + y * y + z * z;
    }

    double rho(double x, double y, double z, double a) {
        return 6 - a * phi(x, y, z);
    }

    void create_grid_communicator(int procNum) {
        MPI_Dims_create(procNum, Coords::TotalDims, dims);

        int periods[Coords::TotalDims] = {false, false};
        int reorder = true;

        MPI_Cart_create(MPI_COMM_WORLD, Coords::TotalDims, dims, periods, reorder, &gridComm);
        MPI_Cart_get(gridComm, Coords::TotalDims, dims, periods, coords);
        MPI_Cart_rank(gridComm, coords, &gridRank);
    }


    DiscreteFunc getPartialFunc(const Domain &domain, const Grid &grid) {
        double stepX = domain.extent.lenX / (grid.nodesX - 1);
        double stepY = domain.extent.lenY / (grid.nodesY - 1);
        double stepZ = domain.extent.lenZ / (grid.nodesZ - 1);

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

        // add shadow edges
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

    void check_result(DiscreteFunc &func) {
        const Domain &domain = func.getDomain();
        const Grid &grid = func.getGrid();
        DiscreteFunc realFunc(domain, grid);
        realFunc.setValues(phi);

    }

} // anonymous namespace

void calculate(int procNum, int nodesX, int nodesY, int nodesZ, double eps) {
    create_grid_communicator(procNum);

    double x0 = -1.0, y0 = -1.0, z0 = -1.0;
    double lenX = 2.0, lenY = 2.0, lenZ = 2.0;
    Domain domain(x0, y0, z0, lenX, lenY, lenZ);
    Grid grid(nodesX, nodesY, nodesZ);
    DiscreteFunc fullFunc = (gridRank == rootRank) ? DiscreteFunc(domain, grid) : DiscreteFunc();

    DiscreteFunc partialFunc = getPartialFunc(domain, grid);
    setEdgeValues(partialFunc);

    if (gridRank == rootRank) {
        check_result(fullFunc);
    }

    MPI_Comm_free(&gridComm);
}
