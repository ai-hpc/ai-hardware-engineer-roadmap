#ifndef COURSE_SPSC_QUEUE_H
#define COURSE_SPSC_QUEUE_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

struct Sample {
    std::uint64_t sequence;
    std::uint64_t payload;
    std::uint64_t checksum;
};

// Exactly one producer calls push; exactly one consumer calls pop.
// The caller owns the queue's lifetime and must join both before destruction.
template <std::size_t Capacity>
class SpscQueue {
    static_assert(Capacity > 0 && Capacity < std::numeric_limits<std::size_t>::max());
    static constexpr std::size_t slots = Capacity + 1;
    std::array<Sample, slots> storage_{};
    std::atomic<std::size_t> write_{0};
    std::atomic<std::size_t> read_{0};
    static std::size_t next(std::size_t index) noexcept {
        return index + 1 == slots ? 0 : index + 1;
    }
public:
    bool push(const Sample& item) noexcept {
        const auto write = write_.load(std::memory_order_relaxed);
        const auto following = next(write);
#ifndef COURSE_MUTATE_SKIP_FULL_CHECK
        if (following == read_.load(std::memory_order_acquire)) return false;
#endif
        storage_[write] = item;
        write_.store(following, std::memory_order_release);
        return true;
    }
    bool pop(Sample& item) noexcept {
        const auto read = read_.load(std::memory_order_relaxed);
        if (read == write_.load(std::memory_order_acquire)) return false;
        item = storage_[read];
        read_.store(next(read), std::memory_order_release);
        return true;
    }
    bool indices_are_lock_free() const noexcept {
        return write_.is_lock_free() && read_.is_lock_free();
    }
};
#endif
