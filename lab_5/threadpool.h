#pragma once

#include <thread>
#include <functional>
#include <future>
#include <cmath>
#include <deque>
#include "task.h"

class ThreadPool final {
public:
    ThreadPool(std::size_t threadsNumber) {
        _workers.reserve(threadsNumber);
        auto worker = std::bind(&ThreadPool::worker, this);
        for (std::size_t threadIx = 0; threadIx < threadsNumber; ++threadIx) {
            _workers.emplace_back(worker);
        }
    }

    ~ThreadPool() {
        for (auto &worker: _workers) {
            worker.join();
        }
    }

    void start() {
        std::scoped_lock<std::mutex> lock{_mutex};
        _isRunning = true;
        _cv.notify_all();
    }

    bool is_running() const {
        return _isRunning;
    }

    void shutdown() {
        std::scoped_lock<std::mutex> lock{_mutex};
        _isRunning = false;
        _cv.notify_all();
    }

    std::future<double> submit(Task &&task) {
        std::scoped_lock<std::mutex> lock{_mutex};
        _tasks.push_back(task);
        _cv.notify_one();

        std::promise<double> promise;
        std::future<double> future = promise.get_future();
        _promises.push_front(std::move(promise));

        return future;
    }

    std::vector<Task> cancel(std::size_t tasksNumber) {
        std::scoped_lock<std::mutex> lock{_mutex};

        auto tasksBeg = _tasks.begin();
        auto tasksToCancel = std::min(tasksNumber, _tasks.size());
        auto tasksEnd = std::next(tasksBeg, static_cast<long>(tasksToCancel));
        std::vector<Task> cancelledTasks(std::make_move_iterator(tasksBeg),
                                         std::make_move_iterator(tasksEnd));
        _tasks.erase(tasksBeg, tasksEnd);
        if (_tasks.empty()) {
            _cvEmpty.notify_all();
        }

        auto promisesBeg = _promises.begin();
        auto promisesEnd = std::next(promisesBeg, static_cast<long>(tasksToCancel));
        _promises.erase(promisesBeg, promisesEnd);

        return cancelledTasks;
    }

    std::size_t tasks_pending() {
        std::scoped_lock<std::mutex> lock{_mutex};
        return _tasks.size();
    }

    void wait_until_empty() {
        std::unique_lock<std::mutex> lock{_mutex};
        while (!_tasks.empty()) {
            _cvEmpty.wait(lock);
        }
    }
private:
    std::deque<Task> _tasks;
    std::deque<std::promise<double>> _promises;
    std::atomic_bool _isRunning = false;
    std::vector<std::thread> _workers;

    std::mutex _mutex;
    std::condition_variable _cv;
    std::condition_variable _cvEmpty;

    void wait_for_start() {
        std::unique_lock<std::mutex> lock{_mutex};
        while (!_isRunning) {
            _cv.wait(lock);
        }
    }

    void worker() {
        wait_for_start();

        Task task;
        std::promise<double> promise;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock{_mutex};

                while (_tasks.empty()) {
                    if (!_isRunning) {
                        return;
                    }
                    _cv.wait(lock);
                }

                task = std::move(_tasks.back());
                _tasks.pop_back();

                promise = std::move(_promises.back());
                _promises.pop_back();

                if (_tasks.empty()) {
                    _cvEmpty.notify_all();
                }
            }

            double result = task();
            promise.set_value(result);
        }
    }
};
