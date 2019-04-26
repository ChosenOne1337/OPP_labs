#ifndef DISCRETEFUNC_H
#define DISCRETEFUNC_H

#include <cstddef>
#include <cstring>
#include <memory>

struct Domain {
    struct Origin {
        double x, y, z;

        Origin(): x(), y(), z() {}

        Origin(double x, double y, double z) {
            this->x = x; this->y = y; this->z = z;
        }
    };

    struct Extent {
        double lenX, lenY, lenZ;

        Extent(): lenX(), lenY(), lenZ() {}

        Extent(double lenX, double lenY, double lenZ) {
            this->lenX = lenX; this->lenY = lenY; this->lenZ = lenZ;
        }
    };

    Origin origin;
    Extent extent;

    Domain() = default;

    Domain(double x0, double y0, double z0, double lenX, double lenY, double lenZ):
        origin(x0, y0, z0), extent(lenX, lenY, lenZ) {}
};

struct Grid {
    int nodesX, nodesY, nodesZ;

    Grid(): nodesX(), nodesY(), nodesZ() {}

    Grid(int nodesX, int nodesY, int nodesZ) {
        this->nodesX = nodesX;
        this->nodesY = nodesY;
        this->nodesZ = nodesZ;
    }
};

class DiscreteFunc {
public:
    friend void swap(DiscreteFunc&, DiscreteFunc&);

    DiscreteFunc() = default;

    ~DiscreteFunc() noexcept {
        if (_data != nullptr) {
            delete[] _data;
            _data = nullptr;
        }

        if (_domain != nullptr) {
            delete _domain;
            _domain = nullptr;
        }

        if (_grid != nullptr) {
            delete _grid;
            _grid = nullptr;
        }
    }

    DiscreteFunc(const Domain &domain, const Grid &grid) {
        int nodesTotal = grid.nodesX * grid.nodesY * grid.nodesZ;
        _data = new double[nodesTotal]();
        _domain = new Domain(domain);
        _grid = new Grid(grid);
        _stepX = _domain->extent.lenX / (_grid->nodesX - 1);
        _stepY = _domain->extent.lenY / (_grid->nodesY - 1);
        _stepZ = _domain->extent.lenZ / (_grid->nodesZ - 1);
    }

    DiscreteFunc(const DiscreteFunc &func): DiscreteFunc(func.getDomain(), func.getGrid()) {
        int nodesTotal = _grid->nodesX * _grid->nodesY * _grid->nodesZ;
        std::memcpy(_data, func.getData(), static_cast<std::size_t>(nodesTotal) * sizeof (double));
    }

    DiscreteFunc(DiscreteFunc &&func) noexcept: DiscreteFunc() {
        swap(*this, func);
    }

    DiscreteFunc& operator= (DiscreteFunc func) noexcept {
        swap(*this, func);
        return *this;
    }

    template<typename Func>
    void setValues(Func func) {
        double value = 0.0;
        double realX = _domain->origin.x;
        double realY = _domain->origin.y;
        double realZ = _domain->origin.z;
        for (int z = 0; z < _grid->nodesZ; ++z, realZ += _stepZ) {
            for (int y = 0; y < _grid->nodesY; ++y, realY += _stepY) {
                for (int x = 0; x < _grid->nodesX; ++x, realX += _stepX) {
                    value = func(realX, realY, realZ);
                    setValue(value, x, y, z);
                }
                realX = _domain->origin.x;
            }
            realY = _domain->origin.y;
        }
    }

    template<typename Func>
    void setTopEdgeValues(Func func) {
        // top XY edge
        double value = 0.0;
        double realX = _domain->origin.x;
        double realY = _domain->origin.y;
        double realZ = _domain->origin.z + _domain->extent.lenZ;
        for (int y = 0; y < _grid->nodesY; ++y, realY += _stepY) {
            for (int x = 0; x < _grid->nodesX; ++x, realX += _stepX) {
                value = func(realX, realY, realZ);
                setValue(value, x, y, _grid->nodesZ - 1);
            }
            realX = _domain->origin.x;
        }
    }

    template<typename Func>
    void setBottomEdgeValues(Func func) {
        // bottom XY edge
        double value = 0.0;
        double realX = _domain->origin.x;
        double realY = _domain->origin.y;
        double realZ = _domain->origin.z;
        for (int y = 0; y < _grid->nodesY; ++y, realY += _stepY) {
            for (int x = 0; x < _grid->nodesX; ++x, realX += _stepX) {
                value = func(realX, realY, realZ);
                setValue(value, x, y, 0);
            }
            realX = _domain->origin.x;
        }
    }

    template<typename Func>
    void setLeftEdgeValues(Func func) {
        // bottom XZ edge
        double value = 0.0;
        double realX = _domain->origin.x;
        double realY = _domain->origin.y;
        double realZ = _domain->origin.z;
        for (int z = 0; z < _grid->nodesZ; ++z, realZ += _stepZ) {
            for (int x = 0; x < _grid->nodesX; ++x, realX += _stepX) {
                value = func(realX, realY, realZ);
                setValue(value, x, 0, z);
            }
            realX = _domain->origin.x;
        }
    }

    template<typename Func>
    void setRightEdgeValues(Func func) {
        // top XZ edge
        double value = 0.0;
        double realX = _domain->origin.x;
        double realY = _domain->origin.y + _domain->extent.lenY;
        double realZ = _domain->origin.z;
        for (int z = 0; z < _grid->nodesZ; ++z, realZ += _stepZ) {
            for (int x = 0; x < _grid->nodesX; ++x, realX += _stepX) {
                value = func(realX, realY, realZ);
                setValue(value, x, _grid->nodesY - 1, z);
            }
            realX = _domain->origin.x;
        }
    }

    template<typename Func>
    void setFrontEdgeValues(Func func) {
        // top YZ edge
        double value = 0.0;
        double realX = _domain->origin.x + _domain->extent.lenX;
        double realY = _domain->origin.y;
        double realZ = _domain->origin.z;
        for (int z = 0; z < _grid->nodesZ; ++z, realZ += _stepZ) {
            for (int y = 0; y < _grid->nodesY; ++y, realY += _stepY) {
                value = func(realX, realY, realZ);
                setValue(value, _grid->nodesX - 1, y, z);
            }
            realY = _domain->origin.y;
        }
    }

    template<typename Func>
    void setBackEdgeValues(Func func) {
        // bottom YZ edge
        double value = 0.0;
        double realX = _domain->origin.x;
        double realY = _domain->origin.y;
        double realZ = _domain->origin.z;
        for (int z = 0; z < _grid->nodesZ; ++z, realZ += _stepZ) {
            for (int y = 0; y < _grid->nodesY; ++y, realY += _stepY) {
                value = func(realX, realY, realZ);
                setValue(value, 0, y, z);
            }
            realY = _domain->origin.y;
        }
    }

    double getValue(int x, int y, int z) const {
        int linearIndex = _get_linear_data_index(x, y, z);
        return _data[linearIndex];
    }

    void setValue(double value, int x, int y, int z) {
        int linearIndex = _get_linear_data_index(x, y, z);
        _data[linearIndex] = value;
    }

    double* getData() {
        return _data;
    }

    const double* getData() const {
        return _data;
    }

    const Grid& getGrid() const {
        return *_grid;
    }

    const Domain& getDomain() const {
        return *_domain;
    }
private:
    double *_data = nullptr;
    Domain *_domain = nullptr;
    Grid *_grid = nullptr;
    double _stepX = 0.0, _stepY = 0.0, _stepZ = 0.0;

    int _get_linear_data_index(int x, int y, int z) const {
        return z * _grid->nodesX * _grid->nodesY + y * _grid->nodesX + x;
    }
};

inline void swap(DiscreteFunc &func1, DiscreteFunc &func2) {
    using std::swap;

    swap(func1._data, func2._data);
    swap(func1._domain, func2._domain);
    swap(func1._grid, func2._grid);

    swap(func1._stepX, func2._stepX);
    swap(func1._stepY, func2._stepY);
    swap(func1._stepZ, func2._stepZ);
}

#endif // DISCRETEFUNC_H
