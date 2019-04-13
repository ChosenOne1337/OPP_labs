#include <mpich/mpi.h>
#include <memory>
#include "calculations.h"

class Domain {
public:
    struct Origin {
        double x, y, z;

        Origin(): x(), y(), z() {}

        Origin(double x, double y, double z) {
            this->x = x; this->y = y; this->z = z;
        }

    };

    Origin origin;

    struct Extent {
        double lenX, lenY, lenZ;

        Extent(): lenX(), lenY(), lenZ() {}

        Extent(double lenX, double lenY, double lenZ) {
            this->lenX = lenX; this->lenY = lenY; this->lenZ = lenZ;
        }

    };

    Extent extent;

    Domain() = default;

    Domain(double x0, double y0, double z0, double lenX, double lenY, double lenZ):
        origin(x0, y0, z0), extent(lenX, lenY, lenZ) {}
};

struct Grid {
    std::size_t nodesX, nodesY, nodesZ;

    Grid(): nodesX(), nodesY(), nodesZ() {}

    Grid(std::size_t nodesX, std::size_t nodesY, std::size_t nodesZ) {
        this->nodesX = nodesX;
        this->nodesY = nodesY;
        this->nodesZ = nodesZ;
    }
};

class DiscreteFunc {
public:
    DiscreteFunc() = default;

    DiscreteFunc(const Domain &domain, const Grid &grid): _domain(domain), _grid(grid) {
        std::size_t nodesTotal = grid.nodesX * grid.nodesY * grid.nodesZ;
        _data.reset(new double[nodesTotal]());
    }

    template<typename Func>
    void setValues(Func func) {
        double stepX = _domain.extent.lenX / (_grid.nodesX - 1);
        double stepY = _domain.extent.lenY / (_grid.nodesY - 1);
        double stepZ = _domain.extent.lenZ / (_grid.nodesZ - 1);

        double realX = _domain.origin.x;
        double realY = _domain.origin.y;
        double realZ = _domain.origin.z;

        double value = 0.0;
        for (std::size_t z = 0; z < _grid.nodesZ; ++z, realZ += stepZ) {
            for (std::size_t y = 0; y < _grid.nodesY; ++y, realY += stepY) {
                for (std::size_t x = 0; x < _grid.nodesX; ++x, realX += stepX) {
                    value = func(realX, realY, realZ);
                    setValue(value, x, y, z);
                }
                realX = _domain.origin.x;
            }
            realY = _domain.origin.y;
        }
    }

    template<typename Func>
    void setEdgeValues(Func func) {
        double stepX = _domain.extent.lenX / (_grid.nodesX - 1);
        double stepY = _domain.extent.lenY / (_grid.nodesY - 1);
        double stepZ = _domain.extent.lenZ / (_grid.nodesZ - 1);

        double realX, realY, realZ;
        double value = 0.0;

        // bottom XY
        realX = _domain.origin.x;
        realY = _domain.origin.y;
        realZ = _domain.origin.z;
        for (std::size_t y = 0; y < _grid.nodesY; ++y, realY += stepY) {
            for (std::size_t x = 0; x < _grid.nodesX; ++x, realX += stepX) {
                value = func(realX, realY, realZ);
                setValue(value, x, y, 0);
            }
            realX = _domain.origin.x;
        }

        // top XY
        realX = _domain.origin.x;
        realY = _domain.origin.y;
        realZ = _domain.origin.z + _domain.extent.lenZ;
        for (std::size_t y = 0; y < _grid.nodesY; ++y, realY += stepY) {
            for (std::size_t x = 0; x < _grid.nodesX; ++x, realX += stepX) {
                value = func(realX, realY, realZ);
                setValue(value, x, y, _grid.nodesZ - 1);
            }
            realX = _domain.origin.x;
        }

        // bottom XZ
        realX = _domain.origin.x;
        realY = _domain.origin.y;
        realZ = _domain.origin.z;
        for (std::size_t z = 0; z < _grid.nodesZ; ++z, realZ += stepZ) {
            for (std::size_t x = 0; x < _grid.nodesX; ++x, realX += stepX) {
                value = func(realX, realY, realZ);
                setValue(value, x, 0, z);
            }
            realX = _domain.origin.x;
        }

        // top XZ
        realX = _domain.origin.x;
        realY = _domain.origin.y + _domain.extent.lenY;
        realZ = _domain.origin.z;
        for (std::size_t z = 0; z < _grid.nodesZ; ++z, realZ += stepZ) {
            for (std::size_t x = 0; x < _grid.nodesX; ++x, realX += stepX) {
                value = func(realX, realY, realZ);
                setValue(value, x, _grid.nodesY - 1, z);
            }
            realX = _domain.origin.x;
        }

        // bottom YZ
        realX = _domain.origin.x;
        realY = _domain.origin.y;
        realZ = _domain.origin.z;
        for (std::size_t z = 0; z < _grid.nodesZ; ++z, realZ += stepZ) {
            for (std::size_t y = 0; y < _grid.nodesY; ++y, realY += stepY) {
                value = func(realX, realY, realZ);
                setValue(value, 0, y, z);
            }
            realY = _domain.origin.y;
        }

        // top YZ
        realX = _domain.origin.x + _domain.extent.lenX;
        realY = _domain.origin.y;
        realZ = _domain.origin.z;
        for (std::size_t z = 0; z < _grid.nodesZ; ++z, realZ += stepZ) {
            for (std::size_t y = 0; y < _grid.nodesY; ++y, realY += stepY) {
                value = func(realX, realY, realZ);
                setValue(value, _grid.nodesX - 1, y, z);
            }
            realY = _domain.origin.y;
        }
    }

    double getValue(std::size_t x, std::size_t y, std::size_t z) const {
        std::size_t linearIndex = _get_linear_data_index(x, y, z);
        return _data[linearIndex];
    }

    void setValue(double value, std::size_t x, std::size_t y, std::size_t z) {
        std::size_t linearIndex = _get_linear_data_index(x, y, z);
        _data[linearIndex] = value;
    }

    double* getRawData() {
        return _data.get();
    }
private:
    std::unique_ptr<double[]> _data;
    Domain _domain = {0., 0., 0., 0., 0., 0.};
    Grid _grid = {0, 0, 0};

    std::size_t _get_linear_data_index(std::size_t x, std::size_t y, std::size_t z) const {
        return z * _grid.nodesX * _grid.nodesY + y * _grid.nodesX + x;
    }
};

namespace Coords {
    enum {
        RowDim,
        ColDim,
        TotalDims
    };
}

namespace {
    int coords[Coords::TotalDims];
    int gridRank;
    int rootRank = 0;
    MPI_Comm gridComm;
}

double phi(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double rho(double x, double y, double z, double a) {
    return 6 - a * phi(x, y, z);
}

void create_grid_communicator(int procNum) {
    int dims[Coords::TotalDims];
    MPI_Dims_create(procNum, Coords::TotalDims, dims);

    int periods[Coords::TotalDims] = {false, false};
    int reorder = true;

    MPI_Cart_create(MPI_COMM_WORLD, Coords::TotalDims, dims, periods, reorder, &gridComm);
    MPI_Cart_get(gridComm, Coords::TotalDims, dims, periods, coords);
    MPI_Cart_rank(gridComm, coords, &gridRank);
}

void calculate(int procNum, std::size_t nodesX, std::size_t nodesY, std::size_t nodesZ, double eps) {
    create_grid_communicator(procNum);

    double x0 = -1.0, y0 = -1.0, z0 = -1.0;
    double lenX = 2.0, lenY = 2.0, lenZ = 2.0;
    Domain domain(x0, y0, z0, lenX, lenY, lenZ);
    Grid grid(nodesX, nodesY, nodesZ);
    DiscreteFunc fullFunc = (gridRank == rootRank) ? DiscreteFunc(domain, grid) : DiscreteFunc();
    if (gridRank == rootRank) {
        fullFunc.setEdgeValues(phi);
    }

    MPI_Comm_free(&gridComm);
}
