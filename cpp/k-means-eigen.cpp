#include <Eigen/Dense>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>


Eigen::MatrixXd k_means(const Eigen::MatrixXd& data,
                        uint16_t k,
                        size_t number_of_iterations) {
  static std::random_device seed;
  static std::mt19937 random_number_generator(seed());
  std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

  Eigen::ArrayX2d means(k, 2);
  for (size_t cluster = 0; cluster < k; ++cluster) {
    means.row(cluster) = data(indices(random_number_generator));
  }

  // Because Eigen does not have native tensors, we'll have to split the data by
  // features and replicate it across columns to reproduce the approach of
  // replicating data across the depth dimension k times.
  const Eigen::ArrayXXd data_x = data.col(0).rowwise().replicate(k);
  const Eigen::ArrayXXd data_y = data.col(1).rowwise().replicate(k);

  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    // This will be optimized nicely by Eigen because it's a large and
    // arithmetic-intense expression tree.
    auto distances = (data_x.rowwise() - means.col(0).transpose()).square() +
                     (data_y.rowwise() - means.col(1).transpose()).square();
    // Unfortunately, Eigen has no vectorized way of retrieving the argmin for
    // every row, so we'll have to loop, and iteratively compute the new
    // centroids.
    Eigen::ArrayX2d sum = Eigen::ArrayX2d::Zero(k, 2);
    Eigen::ArrayXd counts = Eigen::ArrayXd::Ones(k);
    for (size_t index = 0; index < data.rows(); ++index) {
      Eigen::ArrayXd::Index argmin;
      distances.row(index).minCoeff(&argmin);
      sum.row(argmin) += data.row(index).array();
      counts(argmin) += 1;
    }
    means = sum.colwise() / counts;
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

  std::vector<std::pair<float, float>> temporary;
  std::ifstream stream(argv[1]);
  std::string line;
  while (std::getline(stream, line)) {
    float x, y;
    uint16_t label;
    std::istringstream line_stream(line);
    line_stream >> x >> y >> label;
    temporary.emplace_back(x, y);
  }

  Eigen::MatrixXd data(temporary.size(), 2);
  for (size_t row = 0; row < data.rows(); ++row) {
    data.row(row) << temporary[row].first, temporary[row].second;
  }

  Eigen::ArrayX2d means(k, 2);
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

  std::cout << means.format({5, Eigen::DontAlignCols}) << std::endl;
}
