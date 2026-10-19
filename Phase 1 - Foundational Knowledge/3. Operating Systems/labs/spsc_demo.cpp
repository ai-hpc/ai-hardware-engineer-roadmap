#include "spsc_queue.h"
#include <chrono>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <thread>

static Sample sample(std::uint64_t sequence) {
    const auto payload = sequence * UINT64_C(0x9e3779b97f4a7c15);
    return {sequence, payload, sequence ^ payload ^ UINT64_C(0xa5a5a5a5a5a5a5a5)};
}
static bool same(const Sample& a, const Sample& b) {
    return a.sequence == b.sequence && a.payload == b.payload && a.checksum == b.checksum;
}
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template <std::size_t Capacity>
static void sequential_reference() {
    SpscQueue<Capacity> queue;
    Sample output = sample(999);
    require(!queue.pop(output) && same(output, sample(999)), "empty pop modified output");
    for (std::size_t i = 0; i < Capacity; ++i) require(queue.push(sample(i)), "early full");
    require(!queue.push(sample(999)), "full queue accepted overwrite");
    for (std::size_t i = 0; i < Capacity; ++i) {
        require(queue.pop(output) && same(output, sample(i)), "FIFO boundary corruption");
    }
    require(!queue.pop(output), "not empty after drain");
    std::deque<Sample> reference;
    std::uint64_t state = 1234567;
    for (std::uint64_t i = 0; i < 100000; ++i) {
        state = state * UINT64_C(6364136223846793005) + 1;
        if ((state >> 63) != 0) {
            const bool expected = reference.size() < Capacity;
            require(queue.push(sample(i)) == expected, "push/reference disagreement");
            if (expected) reference.push_back(sample(i));
        } else {
            const bool expected = !reference.empty();
            require(queue.pop(output) == expected, "pop/reference disagreement");
            if (expected) {
                require(same(output, reference.front()), "value/reference disagreement");
                reference.pop_front();
            }
        }
    }
    while (!reference.empty()) {
        require(queue.pop(output) && same(output, reference.front()), "final drain mismatch");
        reference.pop_front();
    }
    require(!queue.pop(output), "unexpected item after final drain");
}

template <std::size_t Capacity>
static void concurrent_transfer() {
    SpscQueue<Capacity> queue;
    constexpr std::uint64_t count = 200000;
    std::atomic<bool> failed{false};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::thread producer([&] {
        for (std::uint64_t i = 0; i < count; ++i) {
            while (!queue.push(sample(i))) {
                if (failed.load(std::memory_order_relaxed)) return;
                if (std::chrono::steady_clock::now() >= deadline) {
                    failed.store(true, std::memory_order_relaxed); return;
                }
                std::this_thread::yield();
            }
        }
    });
    try {
        for (std::uint64_t i = 0; i < count; ++i) {
            Sample output{};
            while (!queue.pop(output)) {
                if (failed.load(std::memory_order_relaxed) ||
                    std::chrono::steady_clock::now() >= deadline) {
                    failed.store(true, std::memory_order_relaxed);
                    throw std::runtime_error("concurrent transfer timed out");
                }
                std::this_thread::yield();
            }
            require(same(output, sample(i)), "concurrent order/payload mismatch");
        }
    } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        producer.join();
        throw;
    }
    producer.join();
    require(!failed.load(std::memory_order_relaxed), "producer failed");
    Sample extra{};
    require(!queue.pop(extra), "unexpected concurrent extra item");
}

int main() {
    try {
        sequential_reference<1>();
        sequential_reference<3>();
        sequential_reference<17>();
        concurrent_transfer<1>();
        concurrent_transfer<17>();
        SpscQueue<1> queue;
        std::cout << "{\"sequential_operations\":300000,\"concurrent_items\":400000,"
                  << "\"indices_lock_free_on_this_host\":"
                  << (queue.indices_are_lock_free() ? "true" : "false") << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
