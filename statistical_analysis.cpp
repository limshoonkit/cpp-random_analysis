#include "xoshiro256ss.h"
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// Statistical measures
struct Statistics {
    double mean;
    double stddev;
    double min;
    double max;
    double skewness;
    double kurtosis;
};

Statistics calculate_statistics(const std::vector<double>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double variance = (sq_sum / data.size()) - (mean * mean);
    double stddev = std::sqrt(variance);
    
    double min = *std::min_element(data.begin(), data.end());
    double max = *std::max_element(data.begin(), data.end());
    
    // Calculate skewness
    double skewness = 0.0;
    double kurtosis = 0.0;
    for (double x : data) {
        double diff = (x - mean) / stddev;
        double diff2 = diff * diff;
        skewness += diff * diff2;
        kurtosis += diff2 * diff2;
    }
    skewness = skewness / data.size();
    kurtosis = kurtosis / data.size() - 3.0; // Excess kurtosis (normal = 0)
    
    return {mean, stddev, min, max, skewness, kurtosis};
}

// Create ASCII histogram
void print_histogram(const std::vector<double>& data, int bins = 50) {
    double min = *std::min_element(data.begin(), data.end());
    double max = *std::max_element(data.begin(), data.end());
    std::vector<int> histogram(bins, 0);
    
    for (double value : data) {
        int bin = static_cast<int>((value - min) / (max - min) * (bins - 1));
        if (bin >= 0 && bin < bins) histogram[bin]++;
    }
    
    int max_count = *std::max_element(histogram.begin(), histogram.end());
    const int height = 20;
    
    for (int h = height; h >= 0; --h) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                  << (max_count * h / height) << " |";
        for (int count : histogram) {
            std::cout << (count * height >= max_count * h ? '*' : ' ');
        }
        std::cout << '\n';
    }
    
    std::cout << std::string(9, '-') << '|' << std::string(bins, '-') << '\n';
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << min 
              << " " << std::string(bins/2-5, ' ') << "Value" 
              << std::string(bins/2-5, ' ') << std::setprecision(2) << max << '\n';
}

template<typename RNG>
void analyze_generator(const std::string& name, RNG& gen, size_t samples) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> numbers;
    numbers.reserve(samples);
    
    // Measure generation time
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < samples; ++i) {
        numbers.push_back(dist(gen));
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count() * 1000; // ms
    
    // Calculate statistics
    Statistics stats = calculate_statistics(numbers);
    
    // Print results
    std::cout << "\n=== " << name << " ===\n";
    std::cout << "Time taken: " << std::fixed << std::setprecision(2) << time << " ms\n";
    std::cout << "Mean: " << stats.mean << " (expected: 0)\n";
    std::cout << "StdDev: " << stats.stddev << " (expected: 1)\n";
    std::cout << "Range: [" << stats.min << ", " << stats.max << "]\n";
    std::cout << "Skewness: " << stats.skewness << " (expected: 0)\n";
    std::cout << "Excess Kurtosis: " << stats.kurtosis << " (expected: 0)\n\n";
    
    std::cout << "Distribution:\n";
    print_histogram(numbers);
    std::cout << "\n";
}

int main() {
    const size_t samples = 1000000;
    const uint64_t seed = 42;
    
    std::cout << "Analyzing generators with " << samples << " samples\n";
    
    xoshiro256ss xoshiro(seed);
    std::mt19937_64 mt(seed);
    std::default_random_engine default_gen(seed);
    
    analyze_generator("xoshiro256ss", xoshiro, samples);
    analyze_generator("mt19937_64", mt, samples);
    analyze_generator("default_random", default_gen, samples);
    
    return 0;
}