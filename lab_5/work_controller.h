#pragma once

#include "execution_state.h"
#include "ringcomm.h"
#include "protocol.h"
#include "task.h"

class WorkController final {
public:
    WorkController(RingComm &ringComm, TaskDatatype &taskType):
        _ringComm{ringComm}, _taskType{taskType} {}

    void control_work(ExecutionState &executionState) {
        std::size_t totalTasksCompleted = 0;
        ThreadPool &threadPool = executionState.get_threadpool();
        for (;;) {
            threadPool.wait_until_empty();

            int tasksCompleted = static_cast<int>(executionState.get_tasks_number() - totalTasksCompleted);
            totalTasksCompleted = executionState.get_tasks_number();
            send_message(RingComm::ROOT_RANK, DONE_TAG, &tasksCompleted, 1, MPI_INT);

            for (;;) {
                send_message(_ringComm.get_right_rank(), REQUEST_TAG, nullptr, 0, MPI_INT);
                wait_for_response();
                std::size_t tasksReceived = executionState.get_tasks_number() - totalTasksCompleted;
                if (tasksReceived > 0) {
                    break;
                }

                send_message(_ringComm.get_left_rank(), REQUEST_TAG, nullptr, 0, MPI_INT);
                wait_for_response();
                tasksReceived = executionState.get_tasks_number() - totalTasksCompleted;
                if (tasksReceived > 0) {
                    break;
                }

                wait_for_tasks_available();

                {
                    std::scoped_lock<std::mutex> lock{_mutex};
                    if (_shutdownFlag) {
                        send_end_message();
                        return;
                    }
                }
            }

        }
    }

    void on_received_response() {
        std::scoped_lock<std::mutex> lock{_mutex};
        _responseFlag = true;
        _cv.notify_one();
    }

    void on_tasks_available() {
        std::scoped_lock<std::mutex> lock{_mutex};
        _notifyFlag = true;
        _cv.notify_one();
    }

    void shutdown() {
        std::scoped_lock<std::mutex> lock{_mutex};
        _shutdownFlag = true;
        _cv.notify_one();
    }

private:
    RingComm& _ringComm;
    TaskDatatype& _taskType;

    std::mutex _mutex;
    std::condition_variable _cv;

    bool _shutdownFlag = false;
    bool _notifyFlag = false;
    bool _responseFlag = false;

    void wait_for_response() {
        std::unique_lock<std::mutex> lock{_mutex};
        while (!_shutdownFlag && !_responseFlag) {
            _cv.wait(lock);
        }
        _responseFlag = false;
    }

    void wait_for_tasks_available() {
        std::unique_lock<std::mutex> lock{_mutex};
        while (!_shutdownFlag && !_notifyFlag) {
            _cv.wait(lock);
        }
        _notifyFlag = false;
    }

    void send_message(int procRank, int tag, void *buf, int count, MPI_Datatype datatype) {
        MPI_Send(buf, count, datatype, procRank, tag, _ringComm.get_communicator());
    }

    std::size_t request_tasks(ExecutionState &executionState, int procRank) {
        MPI_Comm &comm = _ringComm.get_communicator();
        MPI_Datatype &taskDatatype = _taskType.get_datatype();

        MPI_Status status;
        send_message(procRank, REQUEST_TAG, nullptr, 0, MPI_INT);
        MPI_Probe(procRank, RESPONSE_TAG, comm, &status);

        int tasksToReceive = 0;
        MPI_Get_count(&status, taskDatatype, &tasksToReceive);
        std::vector<Task> receivedTasks(static_cast<std::size_t>(tasksToReceive));
        MPI_Recv(receivedTasks.data(), tasksToReceive, _taskType.get_datatype(),
                    procRank, RESPONSE_TAG, comm, MPI_STATUS_IGNORE);

        for (auto &task: receivedTasks) {
            executionState.submit_task(std::move(task));
        }

        return receivedTasks.size();
    }

    void notify_neighbors() {
        send_message(_ringComm.get_left_rank(), NOTIFY_TAG, nullptr, 0, MPI_INT);
        if (_ringComm.get_left_rank() != _ringComm.get_right_rank()) {
            send_message(_ringComm.get_right_rank(), NOTIFY_TAG, nullptr, 0, MPI_INT);
        }
    }

    void send_end_message() {
        send_message(_ringComm.get_left_rank(), END_TAG, nullptr, 0, MPI_INT);
        if (_ringComm.get_left_rank() != _ringComm.get_right_rank()) {
            send_message(_ringComm.get_right_rank(), END_TAG, nullptr, 0, MPI_INT);
        }
    }

};
