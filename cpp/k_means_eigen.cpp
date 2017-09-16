#include <Eigen/Dense>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

size_t random_index(size_t upper_bound) {
  static std::mt19937 rng(std::random_device{}());
  static std::uniform_int_distribution<size_t> indices(0, upper_bound - 1);
  return indices(rng);
}

Eigen::MatrixXd k_means(const Eigen::MatrixXd& data,
                        uint16_t k,
                        size_t number_of_iterations) {
  Eigen::MatrixXd means(k, 2);
  for (size_t cluster = 0; cluster < k; ++cluster) {
    means.row(cluster) = data.row(random_index(data.rows()));
  }

  std::vector<uint16_t> assignments(data.rows());
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    for (size_t point = 0; point < data.rows(); ++point) {
      double best_distance = std::numeric_limits<double>::max();
      size_t best_cluster = 0;
      for (size_t cluster = 0; cluster < k; ++cluster) {
        const double distance =
            (means.row(cluster) - data.row(point)).squaredNorm();
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      assignments[point] = best_cluster;
    }

    Eigen::MatrixXd new_means = Eigen::MatrixXd::Zero(k, 2);
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(k);
    for (size_t point = 0; point < data.rows(); ++point) {
      const auto cluster = assignments[point];
      new_means.row(cluster) += data.row(point);
      counts(cluster) += 1;
    }
    means = new_means.array().colwise() / counts.array();
  }

  return means;
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: k_means <data-file> <k> [iterations]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto k = std::atoi(argv[2]);
  const auto iterations = (argc == 4) ? std::atoi(argv[3]) : 300;

  std::vector<std::pair<double, double>> temporary;
  std::ifstream stream(argv[1]);
  std::string line;
  while (std::getline(stream, line)) {
    double x, y;
    uint16_t label;
    std::istringstream line_stream(line);
    line_stream >> x >> y >> label;
    temporary.emplace_back(x, y);
  }

  Eigen::MatrixXd data(temporary.size(), 2);
  for (size_t row = 0; row < data.rows(); ++row) {
    data.row(row) << temporary[row].first, temporary[row].second;
  }

  const auto start = std::chrono::high_resolution_clock::now();
  const auto means = k_means(data, k, iterations);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  std::cerr << "Took: " << duration.count() << "s" << std::endl;

  std::cout << means.format({5, Eigen::DontAlignCols}) << std::endl;
}
