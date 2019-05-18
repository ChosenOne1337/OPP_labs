#pragma once

#include <mpich/mpi.h>
#include "threadpool.h"
#include "ringcomm.h"
#include "protocol.h"
#include "work_controller.h"
#include "request_listener.h"
#include "execution_state.h"

class TaskExecutor final {
public:
    double execute_task(const Task &task, std::size_t threadsNumber) {
        ThreadPool threadPool{threadsNumber};
        std::size_t totalTasksNumber = static_cast<std::size_t>(TASKS_PER_PROCESS * _ringComm.get_size());
        ExecutionState executionState{threadPool, totalTasksNumber};

        auto subtasks = split_task(task);
        for (auto &task: subtasks) {
            executionState.submit_task(std::move(task));
        }

        WorkController workController{_ringComm};
        std::thread workControllerThread {
            std::bind(&WorkController::control_work, &workController, std::ref(executionState))
        };

        RequestListener requestListener{workController, _ringComm, _taskType};
        std::thread requestListenerThread {
            std::bind(&RequestListener::listen_requests, &requestListener, std::ref(executionState))
        };

        threadPool.start();
        workControllerThread.join();
        requestListenerThread.join();
        threadPool.shutdown();

        double localResult = executionState.collect_result();
        double result = get_result(localResult);

        return result;
    }
private:
    RingComm _ringComm;
    TaskDatatype _taskType;

    static constexpr int TASKS_PER_PROCESS = 1000;
    static constexpr double COMPLEXITY_MULTIPLIER = 1.1;

    std::vector<Task> split_task(const Task &task) {
        std::vector<Task> subtasks;
        subtasks.reserve(TASKS_PER_PROCESS);

        double domainExtent = task.rightBoundary - task.leftBoundary;
        double dotsDensity = task.dotsNumber / domainExtent;
        double firstIntervalWidth = get_first_interval_width(domainExtent);
        double intervalWidth = get_interval_width(firstIntervalWidth);
        double subintervalWidth = intervalWidth / TASKS_PER_PROCESS;
        double leftBoundary = get_left_boundary(task.leftBoundary, firstIntervalWidth);

        Task subtask{leftBoundary, leftBoundary, 0};
        subtask.dotsNumber = static_cast<unsigned long long>(dotsDensity * subintervalWidth);

        for (int i = 0; i < TASKS_PER_PROCESS; ++i) {
            subtask.leftBoundary = subtask.rightBoundary;
            subtask.rightBoundary = subtask.leftBoundary + subintervalWidth;
            subtasks.push_back(subtask);
        }

        return subtasks;
    }

    double get_first_interval_width(double domainExtent) {
        double deg = 1.0;
        double sum = 0.0;
        int procNum = _ringComm.get_size();
        for (int i = 0; i < procNum; ++i) {
            sum += deg;
            deg *= COMPLEXITY_MULTIPLIER;
        }
        return domainExtent / sum;
    }

    double get_interval_width(double firstIntervalWidth) {
        int procRank = _ringComm.get_rank();
        double intervalWidth = firstIntervalWidth;
        for (int i = 0; i < procRank; ++i) {
            intervalWidth *= COMPLEXITY_MULTIPLIER;
        }
        return intervalWidth;
    }

    double get_left_boundary(double leftBoundary, double firstIntervalWidth) {
        int procRank = _ringComm.get_rank();
        double procLeftBoundary = leftBoundary;
        double intervalWidth = firstIntervalWidth;
        for (int i = 0; i < procRank; ++i) {
            procLeftBoundary += intervalWidth;
            intervalWidth *= COMPLEXITY_MULTIPLIER;
        }
        return procLeftBoundary;
    }

    double get_result(double localResult) {
        double result = 0.0;
        MPI_Reduce(&localResult, &result, 1, MPI_DOUBLE, MPI_SUM,
                   RingComm::ROOT_RANK, _ringComm.get_communicator());
        return result;
    }
};
