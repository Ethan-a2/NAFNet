#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace {

std::vector<uint16_t> read_input(const std::string& path, size_t expected_elements) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Failed to open input: " + path);
    }
    const auto byte_count = stream.tellg();
    if (byte_count != static_cast<std::streamoff>(expected_elements * sizeof(uint16_t))) {
        throw std::runtime_error("Unexpected input byte count: " + std::to_string(byte_count));
    }
    stream.seekg(0);
    std::vector<uint16_t> data(expected_elements);
    stream.read(reinterpret_cast<char*>(data.data()), byte_count);
    if (!stream) {
        throw std::runtime_error("Failed to read input: " + path);
    }
    return data;
}

void write_output(const std::string& path, const uint16_t* data, size_t element_count) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open output: " + path);
    }
    stream.write(reinterpret_cast<const char*>(data), element_count * sizeof(uint16_t));
    if (!stream) {
        throw std::runtime_error("Failed to write output: " + path);
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7 || argc > 9) {
        std::cerr << "Usage: " << argv[0]
                  << " MODEL INPUT_NCHW_U16 OUTPUT_NCHW_U16 QNN_EP_SO QNN_BACKEND_SO QNN_PROFILE_CSV_OR_DASH [WARMUP] [RUNS]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string output_path = argv[3];
    const std::string qnn_ep_path = argv[4];
    const std::string qnn_backend_path = argv[5];
    const std::string profile_path = argv[6];
    const int warmup_runs = argc >= 8 ? std::stoi(argv[7]) : 3;
    const int measured_runs = argc >= 9 ? std::stoi(argv[8]) : 20;
    if (warmup_runs < 0 || measured_runs < 1) {
        throw std::invalid_argument("WARMUP must be >= 0 and RUNS must be >= 1");
    }

    constexpr int64_t batch = 1;
    constexpr int64_t channels = 3;
    constexpr int64_t height = 360;
    constexpr int64_t width = 640;
    constexpr size_t element_count = batch * channels * height * width;
    const std::array<int64_t, 4> input_shape = {batch, channels, height, width};
    std::vector<uint16_t> input = read_input(input_path, element_count);

    try {
        const auto environment_start = std::chrono::steady_clock::now();
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "nafnet-ort-qnn");
        environment.RegisterExecutionProviderLibrary("QNNExecutionProvider", qnn_ep_path);
        const double environment_ms = elapsed_ms(environment_start);

        std::vector<Ort::ConstEpDevice> selected_devices;
        for (const auto& device : environment.GetEpDevices()) {
            std::cout << "EP device: " << device.EpName() << "\n";
            if (std::strcmp(device.EpName(), "QNNExecutionProvider") == 0) {
                selected_devices.push_back(device);
            }
        }
        if (selected_devices.empty()) {
            throw std::runtime_error("QNNExecutionProvider device not found");
        }

        std::vector<double> run_times_ms;
        run_times_ms.reserve(measured_runs);
        double session_init_ms = 0.0;
        uint16_t output_min = 0;
        uint16_t output_max = 0;

        {
            Ort::SessionOptions session_options;
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_options.SetLogSeverityLevel(2);
            session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

            std::unordered_map<std::string, std::string> provider_options = {
                {"backend_path", qnn_backend_path},
                {"skip_qnn_version_check", "1"},
                {"htp_performance_mode", "burst"},
                {"htp_graph_finalization_optimization_mode", "3"},
                {"vtcm_mb", "8"},
                {"qnn_context_priority", "high"},
                {"enable_htp_shared_memory_allocator", "1"},
                {"offload_graph_io_quantization", "0"},
            };
            if (profile_path != "-") {
                provider_options.emplace("profiling_level", "basic");
                provider_options.emplace("profiling_file_path", profile_path);
            }
            session_options.AppendExecutionProvider_V2(environment, selected_devices, provider_options);

            const auto session_start = std::chrono::steady_clock::now();
            Ort::Session session(environment, model_path.c_str(), session_options);
            session_init_ms = elapsed_ms(session_start);

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<uint16_t>(
                memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());
            const char* input_names[] = {"image"};
            const char* output_names[] = {"deblurred_image"};
            Ort::RunOptions run_options;
            run_options.AddConfigEntry("qnn.perf_mode", "burst");
            run_options.AddConfigEntry("qnn.rpc_control_latency", "100");

            auto run_once = [&]() {
                return session.Run(
                    run_options, input_names, &input_tensor, 1, output_names, 1);
            };

            for (int run_index = 0; run_index < warmup_runs; ++run_index) {
                run_once();
            }

            std::vector<Ort::Value> output_values;
            for (int run_index = 0; run_index < measured_runs; ++run_index) {
                const auto run_start = std::chrono::steady_clock::now();
                output_values = run_once();
                const double duration_ms = elapsed_ms(run_start);
                run_times_ms.push_back(duration_ms);
                std::cout << "run " << (run_index + 1) << ": " << std::fixed << std::setprecision(3)
                          << duration_ms << " ms\n";
            }

            if (output_values.size() != 1 || !output_values[0].IsTensor()) {
                throw std::runtime_error("Unexpected output count or type");
            }
            const auto output_info = output_values[0].GetTensorTypeAndShapeInfo();
            if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
                output_info.GetElementCount() != element_count) {
                throw std::runtime_error("Unexpected output shape or type");
            }
            const uint16_t* output_data = output_values[0].GetTensorData<uint16_t>();
            const auto minmax = std::minmax_element(output_data, output_data + element_count);
            output_min = *minmax.first;
            output_max = *minmax.second;
            write_output(output_path, output_data, element_count);
        }

        environment.UnregisterExecutionProviderLibrary("QNNExecutionProvider");

        std::vector<double> sorted_times = run_times_ms;
        std::sort(sorted_times.begin(), sorted_times.end());
        const double average_ms = std::accumulate(run_times_ms.begin(), run_times_ms.end(), 0.0) /
                                  static_cast<double>(run_times_ms.size());
        const double median_ms = sorted_times.size() % 2 == 0
                                     ? (sorted_times[sorted_times.size() / 2 - 1] +
                                        sorted_times[sorted_times.size() / 2]) /
                                           2.0
                                     : sorted_times[sorted_times.size() / 2];

        std::cout << std::fixed << std::setprecision(3)
                  << "environment_and_plugin_init_ms=" << environment_ms << "\n"
                  << "session_init_ms=" << session_init_ms << "\n"
                  << "average_ms=" << average_ms << "\n"
                  << "median_ms=" << median_ms << "\n"
                  << "minimum_ms=" << sorted_times.front() << "\n"
                  << "maximum_ms=" << sorted_times.back() << "\n"
                  << "output_native_min=" << output_min << "\n"
                  << "output_native_max=" << output_max << "\n";
    } catch (const Ort::Exception& exception) {
        std::cerr << "ONNX Runtime error: " << exception.what() << "\n";
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        return 1;
    }

    return 0;
}
