// Small executable contracts, not performance or real-time measurements.
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_hash_map.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace tbb = oneapi::tbb;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void queue_contract() {
    tbb::concurrent_bounded_queue<int> queue;
    queue.set_capacity(2);  // The constructor does not accept an integer capacity.
    require(queue.try_push(10) && queue.try_push(20), "queue fill");
    require(!queue.try_push(30), "full queue must reject try_push");
    int value = -1;
    require(queue.try_pop(value) && value == 10, "queue FIFO first");
    require(queue.try_push(30), "released capacity");
    require(queue.try_pop(value) && value == 20, "queue FIFO second");
    require(queue.try_pop(value) && value == 30, "queue FIFO third");
    require(!queue.try_pop(value), "empty queue must reject try_pop");
    std::cout << "PASS bounded queue: capacity and FIFO\n";
}

void container_contract() {
    constexpr int count = 24000;
    tbb::concurrent_hash_map<int, int> frequencies;
    tbb::parallel_for(0, count, [&](int index) {
        tbb::concurrent_hash_map<int, int>::accessor entry;
        frequencies.insert(entry, index % 17);
        ++entry->second;  // Exclusive element access lasts through this update.
    });
    for (int key = 0; key < 17; ++key) {
        tbb::concurrent_hash_map<int, int>::const_accessor entry;
        require(frequencies.find(entry, key), "missing key");
        int expected = 0;
        for (int index = key; index < count; index += 17) ++expected;
        require(entry->second == expected, "lost protected update");
    }
    tbb::concurrent_unordered_map<int, int> immutable_values;
    tbb::parallel_for(0, count, [&](int index) {
        immutable_values.emplace(index % 257, (index % 257) * 3);
    });
    require(immutable_values.size() == 257, "unique insertion count");
    for (const auto& entry : immutable_values)
        require(entry.second == entry.first * 3, "immutable inserted value");
    std::cout << "PASS containers: 24000 protected updates and duplicate inserts\n";
}

void nested_ownership_contract(bool isolate) {
    constexpr int rows = 16, columns = 64;
    std::vector<int> data(rows * columns, -1);
    tbb::parallel_for(0, rows, [&](int row) {
        auto inner = [&] {
            tbb::parallel_for(0, columns, [&](int column) {
                data[row * columns + column] = row * row + column;
            });
        };
        if (isolate) tbb::this_task_arena::isolate(inner);
        else inner();
    });
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
            require(data[row * columns + column] == row * row + column,
                    "disjoint nested writes");
    std::cout << "PASS disjoint nested writes: isolate=" << isolate << '\n';
}

void isolation_is_not_mutex() {
    // Real overlapping regions, with all shared state protected by this mutex.
    // The mutex is released by condition_variable::wait_for while waiting.
    std::mutex mutex;
    std::condition_variable condition;
    int entered = 0, active = 0, max_active = 0;
    bool timed_out = false;
    auto participant = [&] {
        tbb::this_task_arena::isolate([&] {
            std::unique_lock<std::mutex> lock(mutex);
            ++entered;
            ++active;
            max_active = std::max(max_active, active);
            condition.notify_all();
            if (!condition.wait_for(lock, std::chrono::seconds(5), [&] { return entered == 2; }))
                timed_out = true;
            --active;
        });
    };
    std::thread first(participant), second(participant);
    first.join();
    second.join();
    require(!timed_out && max_active == 2, "isolated regions did not overlap");
    std::cout << "PASS isolate is not a mutex: concurrent regions=2\n";
}

void cancellation_contract() {
    // One calling application thread + disabled workers makes this boundary
    // example easy to inspect. It is not a multiworker cancellation-latency test.
    tbb::global_control sequential(tbb::global_control::max_allowed_parallelism, 1);
    tbb::task_group_context context;
    int committed = 0;
    bool first_request = false, repeated_request = true;
    tbb::parallel_for(tbb::blocked_range<int>(0, 100, 100),
        [&](const tbb::blocked_range<int>& range) {
            for (int index = range.begin(); index != range.end(); ++index) {
                if (context.is_group_execution_cancelled()) return;
                ++committed;
                first_request = context.cancel_group_execution();
                repeated_request = context.cancel_group_execution();
                // The next iteration cooperates. The first write stays committed.
            }
        }, tbb::simple_partitioner{}, context);
    require(committed == 1 && first_request && !repeated_request,
            "cooperative cancellation or no-rollback contract");
    std::cout << "PASS cancellation: one committed write, repeated request=false\n";
}

int main(int argc, char** argv) {
    if (argc > 1 && (argc != 2 || std::string(argv[1]) != "--self-test")) {
        std::cerr << "usage: concurrency_contracts [--self-test]\n";
        return 2;
    }
    try {
        tbb::global_control budget(tbb::global_control::max_allowed_parallelism, 4);
        queue_contract();
        container_contract();
        nested_ownership_contract(false);
        nested_ownership_contract(true);
        isolation_is_not_mutex();
        cancellation_contract();
        std::cout << "PASS all 6 concurrency contracts\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
