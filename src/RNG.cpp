#include "RNG.hpp"

std::mt19937 RNG::gen(std::random_device{}());
std::uniform_real_distribution<double> RNG::dist(0.0, 1.0);

std::mt19937& RNG::get_engine()
{
    return gen;
}

void RNG::set_seed(int seed)
{
    gen.seed(seed);
}

double RNG::get_random()
{
    return dist(gen);
}

double RNG::get_random_range(double min, double max)
{
    std::uniform_real_distribution<double> range_dist(min, max);
    return range_dist(gen);
}

double RNG::get_normal(double mean, double stddev)
{
    std::normal_distribution<double> normal_dist(mean, stddev);
    return normal_dist(gen);
}
