// Synthetic image-feature pipeline; no camera, model, or GPU is used.
// g++ -O2 -std=c++17 -pthread video_pipeline_template.cpp -ltbb -o video_pipeline
// ./video_pipeline [frames=20] [max_inflight=4] [threads=2]
// ./video_pipeline --self-test
// source -> limiter -> accounting -> broadcast -> sum/max branches
//        -> key-matching join -> combine -> sequencer -> output -> decrementer
#include <oneapi/tbb/flow_graph.h>
#include <oneapi/tbb/global_control.h>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

using namespace oneapi::tbb::flow;
struct Frame { int id; std::shared_ptr<const std::vector<float>> pixels; };
struct Feature { int id; float value; };
using Features = std::tuple<Feature, Feature>;

int number(const char* argument, int minimum, int maximum) {
    std::string_view text(argument);
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        value < minimum || value > maximum) throw std::invalid_argument("invalid integer argument");
    return value;
}
void verify_join_contract() {
    graph g;
    join_node<Features, key_matching<int>> matched(g,
        [](const Feature& f) { return f.id; }, [](const Feature& f) { return f.id; });
    join_node<Features, queueing> queued(g);
    int good = 0, bad = 0;
    function_node<Features, continue_msg> check_match(g, serial, [&](const Features& pair) {
        if (std::get<0>(pair).id != std::get<1>(pair).id)
            throw std::runtime_error("key-matching joined different frames");
        ++good;
        return continue_msg{};
    });
    function_node<Features, continue_msg> check_queue(g, serial, [&](const Features& pair) {
        if (std::get<0>(pair).id != std::get<1>(pair).id) ++bad;
        return continue_msg{};
    });
    make_edge(matched, check_match);
    make_edge(queued, check_queue);
    // Deliberately reversed branch-completion order, independent of scheduling.
    input_port<0>(matched).try_put({0, 0}); input_port<0>(matched).try_put({1, 1});
    input_port<1>(matched).try_put({1, 1}); input_port<1>(matched).try_put({0, 0});
    input_port<0>(queued).try_put({0, 0}); input_port<0>(queued).try_put({1, 1});
    input_port<1>(queued).try_put({1, 1}); input_port<1>(queued).try_put({0, 0});
    g.wait_for_all();
    if (good != 2 || bad != 2) throw std::runtime_error("join fault experiment did not detect expected mismatch");
}
void run_pipeline(int count, int limit) {
    graph g;
    int generated = 0, delivered = 0;
    std::atomic<int> live{0}, peak{0};
    input_node<Frame> source(g, [&](oneapi::tbb::flow_control& control) -> Frame {
        if (generated == count) { control.stop(); return {}; }
        const int id = generated++;
        return {id, std::make_shared<const std::vector<float>>(64, static_cast<float>(id))};
    });
    limiter_node<Frame> limiter(g, limit);
    function_node<Frame, Frame> account(g, unlimited, [&](Frame frame) {
        const int current = live.fetch_add(1) + 1;
        int previous = peak.load();
        while (previous < current && !peak.compare_exchange_weak(previous, current)) {}
        if (current > limit) throw std::runtime_error("in-flight bound exceeded");
        return frame;
    });
    broadcast_node<Frame> broadcast(g);
    function_node<Frame, Feature> sum(g, unlimited, [](const Frame& frame) {
        return Feature{frame.id, std::accumulate(frame.pixels->begin(), frame.pixels->end(), 0.0f)};
    });
    function_node<Frame, Feature> maximum(g, unlimited, [](const Frame& frame) {
        return Feature{frame.id, *std::max_element(frame.pixels->begin(), frame.pixels->end())};
    });
    join_node<Features, key_matching<int>> join(g,
        [](const Feature& f) { return f.id; }, [](const Feature& f) { return f.id; });
    function_node<Features, Feature> combine(g, unlimited, [](const Features& pair) {
        const auto& a = std::get<0>(pair);
        const auto& b = std::get<1>(pair);
        if (a.id != b.id) throw std::runtime_error("frame identity mismatch");
        return Feature{a.id, a.value + b.value};
    });
    sequencer_node<Feature> sequencer(g, [](const Feature& f) { return static_cast<std::size_t>(f.id); });
    function_node<Feature, continue_msg> output(g, serial, [&](const Feature& result) {
        if (result.id != delivered || result.value != 65.0f * result.id)
            throw std::runtime_error("order or numerical-reference mismatch");
        ++delivered;
        live.fetch_sub(1);
        return continue_msg{};
    });
    make_edge(source, limiter);
    make_edge(limiter, account);
    make_edge(account, broadcast);
    make_edge(broadcast, sum);
    make_edge(broadcast, maximum);
    make_edge(sum, input_port<0>(join));
    make_edge(maximum, input_port<1>(join));
    make_edge(join, combine);
    make_edge(combine, sequencer);
    make_edge(sequencer, output);
    make_edge(output, limiter.decrementer());
    source.activate();
    g.wait_for_all();
    if (generated != count || delivered != count || live.load() != 0 || peak.load() > limit)
        throw std::runtime_error("final accounting mismatch");
    std::printf("PASS frames=%d delivered=%d peak_inflight=%d limit=%d\n", count, delivered, peak.load(), limit);
    // Bound is accepted logical frames, not total resident bytes. The input node
    // may cache one pending frame; graph bookkeeping and shared_ptr copies remain.
}
int main(int argc, char** argv) {
    try {
        const bool self_test = argc == 2 && std::string_view(argv[1]) == "--self-test";
        if (!self_test && argc > 4) throw std::invalid_argument("too many arguments");
        const int frames = !self_test && argc > 1 ? number(argv[1], 0, 10000) : 20;
        const int limit = !self_test && argc > 2 ? number(argv[2], 1, 10000) : 4;
        const int threads = !self_test && argc > 3 ? number(argv[3], 1, 64) : 2;
        oneapi::tbb::global_control control(oneapi::tbb::global_control::max_allowed_parallelism, threads);
        if (self_test) {
            verify_join_contract();
            for (int count : {0, 1, 17, 100}) for (int bound : {1, 4}) run_pipeline(count, bound);
            std::puts("PASS keyed identity, deliberate FIFO-join mismatch, bounds, ordering, and payload checks");
        } else run_pipeline(frames, limit);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\nUsage: video_pipeline [frames:0..10000] [max_inflight:1..10000] [threads:1..64]\n", error.what());
        return 1;
    }
}
