#pragma once

#include "protocol.h"
#include "ringcomm.h"
#include "task.h"
#include "execution_state.h"
#include "work_controller.h"

class RequestListener final {
public:
    RequestListener(WorkController &workController, RingComm &ringComm, TaskDatatype &taskType):
        _ringComm{ringComm}, _taskType{taskType}, _workController{workController} {}

    void listen_requests(ExecutionState &executionState) {
        while (_isRunning || _leftIsRunning || _rightIsRunning) {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, _ringComm.get_communicator(), &status);

            int procRank = status.MPI_SOURCE;
            int tag = status.MPI_TAG;

            switch (tag) {
            case REQUEST_TAG:
                handle_request(executionState, procRank);
                break;
            case RESPONSE_TAG:
                handle_response(executionState, procRank);
                break;
            case NOTIFY_TAG:
                handle_notify_message(procRank);
                break;
            case SHUTDOWN_TAG:
                handle_shutdown_message(procRank);
                break;
            case DONE_TAG:
                handle_done_message(executionState, procRank);
                break;
            case END_TAG:
                handle_end_message(procRank);
                break;
            default:
                break;
            }

        }
    }

private:
    RingComm& _ringComm;
    TaskDatatype& _taskType;
    WorkController &_workController;

    bool _isRunning = true;
    bool _leftIsRunning = true;
    bool _rightIsRunning = true;
    std::size_t _totalTasksCompleted = 0;

    void receive_message(int procRank, int tag, void *buf, int count, MPI_Datatype datatype) {
        MPI_Recv(buf, count, datatype, procRank, tag, _ringComm.get_communicator(), MPI_STATUS_IGNORE);
    }

    void send_message(int procRank, int tag, void *buf, int count, MPI_Datatype datatype) {
        MPI_Request request;
        MPI_Isend(buf, count, datatype, procRank, tag, _ringComm.get_communicator(), &request);
        MPI_Request_free(&request);
    }

    void handle_request(ExecutionState &executionState, int procRank) {
        receive_message(procRank, REQUEST_TAG, nullptr, 0, MPI_INT);

        auto tasksPending = executionState.get_threadpool().tasks_pending();
        auto tasks = executionState.cancel_tasks(tasksPending / 2);

        send_message(procRank, RESPONSE_TAG, tasks.data(), static_cast<int>(tasks.size()), _taskType.get_datatype());
    }

    void handle_response(ExecutionState &executionState, int procRank) {
        MPI_Status status;
        MPI_Probe(procRank, RESPONSE_TAG, _ringComm.get_communicator(), &status);

        int tasksToReceive = 0;
        MPI_Get_count(&status, _taskType.get_datatype(), &tasksToReceive);
        std::vector<Task> receivedTasks(static_cast<std::size_t>(tasksToReceive));
        receive_message(procRank, RESPONSE_TAG, receivedTasks.data(), tasksToReceive, _taskType.get_datatype());

        for (auto &task: receivedTasks) {
            executionState.submit_task(std::move(task));
        }

        _workController.on_received_response();

        if (executionState.get_threadpool().tasks_pending() > 0) {
            notify_neighbors();
        }
    }

    void notify_neighbors() {
        send_message(_ringComm.get_left_rank(), NOTIFY_TAG, nullptr, 0, MPI_INT);
        if (_ringComm.get_left_rank() != _ringComm.get_right_rank()) {
            send_message(_ringComm.get_right_rank(), NOTIFY_TAG, nullptr, 0, MPI_INT);
        }
    }

    void handle_notify_message(int procRank) {
        receive_message(procRank, NOTIFY_TAG, nullptr, 0, MPI_INT);
        _workController.on_tasks_available();
    }

    void handle_shutdown_message(int procRank) {
        receive_message(procRank, SHUTDOWN_TAG, nullptr, 0, MPI_INT);
        _workController.shutdown();
        _isRunning = false;
    }

    void handle_end_message(int procRank) {
        receive_message(procRank, END_TAG, nullptr, 0, MPI_INT);

        if (procRank == _ringComm.get_left_rank()) {
            _leftIsRunning = false;
        }

        if (procRank == _ringComm.get_right_rank()) {
            _rightIsRunning = false;
        }
    }

    void handle_done_message(ExecutionState &executionState, int procRank) {
        int tasksDone = 0;
        receive_message(procRank, DONE_TAG, &tasksDone, 1, MPI_INT);

        _totalTasksCompleted += static_cast<std::size_t>(tasksDone);
        if (_totalTasksCompleted == executionState.get_total_tasks_number()) {
            broadcast_shutdown_message();
        }
    }

    void broadcast_shutdown_message() {
        int procNum = _ringComm.get_size();
        for (int procRank = 0; procRank < procNum; ++procRank) {
            send_message(procRank, SHUTDOWN_TAG, nullptr, 0, MPI_INT);
        }
    }

};
