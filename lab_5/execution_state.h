#pragma once

#include <cstddef>
#include <deque>
#include <future>
#include "threadpool.h"

class ExecutionState final {
public:
    ExecutionState(ThreadPool &threadPool, std::size_t totalTasksNumber):
        _threadPool{threadPool}, _totalTasksNumber{totalTasksNumber} {}

    ThreadPool& get_threadpool() {
        return _threadPool;
    }

    std::size_t get_tasks_number() {
        std::scoped_lock<std::mutex> lock{_mutex};
        return _futures.size();
    }

    std::size_t get_total_tasks_number() const {
        return _totalTasksNumber;
    }

    void submit_task(Task &&task) {
        std::scoped_lock<std::mutex> lock{_mutex};
        std::future<double> &&future = _threadPool.submit(std::move(task));
        _futures.emplace_front(std::move(future));
    }

    std::vector<Task> cancel_tasks(std::size_t tasksToCancel) {
        std::scoped_lock<std::mutex> lock{_mutex};
        auto &&cancelledTasks = _threadPool.cancel(tasksToCancel);
        std::size_t tasksCancelled = cancelledTasks.size();

        auto promisesBeg = _futures.begin();
        auto promisesEnd = std::next(promisesBeg, static_cast<long>(tasksCancelled));
        _futures.erase(promisesBeg, promisesEnd);

        return std::move(cancelledTasks);
    }

    double collect_result() {
        double result = 0.0;
        for (auto &future: _futures) {
            result += future.get();
        }
        return result;
    }
private:
    std::mutex _mutex;
    std::deque<std::shared_future<double>> _futures;

    ThreadPool &_threadPool;
    const std::size_t _totalTasksNumber;
};
