#pragma once
#include <random>

class RNG
{
    private:
        static std::mt19937 gen;
        static std::uniform_real_distribution<double> dist;

    public:
        // Core generator access
        static std::mt19937& get_engine();
        static void set_seed(int seed);

        // Uniform distribution [0, 1]
        static double get_random();
        
        // Uniform distribution [min, max]
        static double get_random_range(double min, double max);

        // Normal distribution
        static double get_normal(double mean, double stddev);
};