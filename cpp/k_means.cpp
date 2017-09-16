#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

struct Point {
  double x{0}, y{0};
};

using DataFrame = std::vector<Point>;

double square(double value) {
  return value * value;
}

double squared_l2_distance(Point first, Point second) {
  return square(first.x - second.x) + square(first.y - second.y);
}

DataFrame k_means(const DataFrame& data,
                  size_t k,
                  size_t number_of_iterations) {
  static std::random_device seed;
  static std::mt19937 random_number_generator(seed());
  std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

  // Pick centroids as random points from the dataset.
  DataFrame means(k);
  for (auto& cluster : means) {
    cluster = data[indices(random_number_generator)];
  }

  std::vector<size_t> assignments(data.size());
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    // Find assignments.
    for (size_t point = 0; point < data.size(); ++point) {
      double best_distance = std::numeric_limits<double>::max();
      size_t best_cluster = 0;
      for (size_t cluster = 0; cluster < k; ++cluster) {
        const double distance =
            squared_l2_distance(data[point], means[cluster]);
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      assignments[point] = best_cluster;
    }

    // Sum up and count points for each cluster.
    DataFrame new_means(k);
    std::vector<size_t> counts(k, 0);
    for (size_t point = 0; point < data.size(); ++point) {
      const auto cluster = assignments[point];
      new_means[cluster].x += data[point].x;
      new_means[cluster].y += data[point].y;
      counts[cluster] += 1;
    }

    // Divide sums by counts to get new centroids.
    for (size_t cluster = 0; cluster < k; ++cluster) {
      // Turn 0/0 into 0/1 to avoid zero division.
      const auto count = std::max<size_t>(1, counts[cluster]);
      means[cluster].x = new_means[cluster].x / count;
      means[cluster].y = new_means[cluster].y / count;
    }
  }

  return means;
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: k_means <data-file> <k> [iterations] [runs]"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto k = std::atoi(argv[2]);
  const auto iterations = (argc >= 4) ? std::atoi(argv[3]) : 300;
  const auto number_of_runs = (argc >= 5) ? std::atoi(argv[4]) : 10;

  DataFrame data;
  std::ifstream stream(argv[1]);
  if (!stream) {
    std::cerr << "Could not open file: " << argv[1] << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(stream, line)) {
    Point point;
    std::istringstream line_stream(line);
    size_t label;
    line_stream >> point.x >> point.y >> label;
    data.push_back(point);
  }

  DataFrame means;
  double total_elapsed = 0;
  for (int run = 0; run < number_of_runs; ++run) {
    const auto start = std::chrono::high_resolution_clock::now();
    means = k_means(data, k, iterations);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    total_elapsed += duration.count();
  }
  std::cerr << "Took: " << total_elapsed / number_of_runs << "s ("
            << number_of_runs << " runs)" << std::endl;

  for (auto& mean : means) {
    std::cout << mean.x << " " << mean.y << std::endl;
  }
}
