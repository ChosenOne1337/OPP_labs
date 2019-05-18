#pragma once

#include <mpich/mpi.h>

class RingComm {
public:
    static constexpr int ROOT_RANK = 0;

    RingComm(const RingComm&) = delete;
    RingComm(RingComm&&) = delete;
    RingComm& operator=(RingComm) = delete;

    RingComm() {
        const int NDIMS = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &procNum);
        int dims[NDIMS] = {procNum};
        int periods[NDIMS] = {true};
        int reorder = true;
        MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &ringComm);

        int displ = 1;
        int direction = 0;
        MPI_Comm_rank(ringComm, &procRank);
        MPI_Cart_shift(ringComm, direction, displ, &leftRank, &rightRank);
    }

    ~RingComm() noexcept {
        if (ringComm != MPI_COMM_NULL) {
            MPI_Comm_free(&ringComm);
            ringComm = MPI_COMM_NULL;
        }
    }

    bool is_root_process() const {
        return procRank == ROOT_RANK;
    }

    int get_size() const {
        return procNum;
    }

    int get_left_rank() const {
        return leftRank;
    }

    int get_right_rank() const {
        return rightRank;
    }

    int get_rank() const {
        return procRank;
    }

    MPI_Comm& get_communicator() {
        return ringComm;
    }
private:
    int procRank;
    int leftRank;
    int rightRank;
    int procNum;
    MPI_Comm ringComm = MPI_COMM_NULL;
};
