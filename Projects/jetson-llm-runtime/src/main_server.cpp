// main_server.cpp — HTTP server entry point
// Usage: jetson-llm-server -m model.gguf [-p port] [-c context]

#include "jllm.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <signal.h>

namespace jllm {
void run_server(Engine& engine, int port);
}

static jllm::Engine* g_engine = nullptr;
void signal_handler(int) {
    if (g_engine) g_engine->stop();
    fprintf(stderr, "\n[server] shutting down\n");
    exit(0);
}

int main(int argc, char** argv) {
    std::string model_path;
    int port = 8080;
    int context = 0;
    bool kv_int8 = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i+1 < argc) context = atoi(argv[++i]);
        else if (strcmp(argv[i], "--fp16-kv") == 0) kv_int8 = false;
        else if (strcmp(argv[i], "-h") == 0) {
            fprintf(stderr,
                "jetson-llm-server — OpenAI-compatible LLM API for Jetson\n\n"
                "Usage: jetson-llm-server -m model.gguf [options]\n\n"
                "  -m PATH      GGUF model (required)\n"
                "  -p PORT      HTTP port (default: 8080)\n"
                "  -c INT       Context length (0 = auto from memory)\n"
                "  --fp16-kv    Use FP16 KV cache (the supported teaching default)\n\n"
                "Endpoints:\n"
                "  GET  /health                  Jetson system health\n"
                "  GET  /v1/models               List loaded model\n"
                "  POST /v1/chat/completions     OpenAI-compatible chat\n\n"
                "Example:\n"
                "  curl http://jetson:8080/v1/chat/completions \\\n"
                "    -H 'Content-Type: application/json' \\\n"
                "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}'\n");
            return 0;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: -m model.gguf required\n");
        return 1;
    }

    // System probe
    auto info = jllm::probe_jetson();
    jllm::print_jetson_info(info);

    // Load model
    jllm::Engine engine;
    g_engine = &engine;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    jllm::GenParams params;
    params.context_limit = context;
    params.kv_int8 = kv_int8;

    fprintf(stderr, "Loading model: %s\n", model_path.c_str());
    if (!engine.load(model_path, params)) {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }

    // Start server
    jllm::run_server(engine, port);
    return 0;
}
