#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__device__ float
squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

// In the assignment step, each point (thread) computes its distance to each
// cluster centroid and adds its x and y values to the sum of its closest
// centroid, as well as incrementing that centroid's count of assigned points.
__global__ void assign_clusters(const thrust::device_ptr<float> data_x,
                                const thrust::device_ptr<float> data_y,
                                int data_size,
                                const thrust::device_ptr<float> means_x,
                                const thrust::device_ptr<float> means_y,
                                thrust::device_ptr<float> new_sums_x,
                                thrust::device_ptr<float> new_sums_y,
                                int k,
                                thrust::device_ptr<int> counts) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= data_size) return;

  // Make global loads once.
  const float x = data_x[index];
  const float y = data_y[index];

  float best_distance = FLT_MAX;
  int best_cluster = 0;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance =
        squared_l2_distance(x, y, means_x[cluster], means_y[cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  atomicAdd(thrust::raw_pointer_cast(new_sums_x + best_cluster), x);
  atomicAdd(thrust::raw_pointer_cast(new_sums_y + best_cluster), y);
  atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(thrust::device_ptr<float> means_x,
                                  thrust::device_ptr<float> means_y,
                                  const thrust::device_ptr<float> new_sum_x,
                                  const thrust::device_ptr<float> new_sum_y,
                                  const thrust::device_ptr<int> counts) {
  const int cluster = threadIdx.x;
  const int count = max(1, counts[cluster]);
  means_x[cluster] = new_sum_x[cluster] / count;
  means_y[cluster] = new_sum_y[cluster] / count;
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: assign_clusters <data-file> <k> [iterations]"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto k = std::atoi(argv[2]);
  const auto number_of_iterations = (argc == 4) ? std::atoi(argv[3]) : 300;

  thrust::host_vector<float> h_x;
  thrust::host_vector<float> h_y;
  std::ifstream stream(argv[1]);
  std::string line;
  while (std::getline(stream, line)) {
    std::istringstream line_stream(line);
    float x, y;
    uint16_t label;
    line_stream >> x >> y >> label;
    h_x.push_back(x);
    h_y.push_back(y);
  }

  const size_t number_of_elements = h_x.size();

  thrust::device_vector<float> d_x = h_x;
  thrust::device_vector<float> d_y = h_y;

  std::mt19937 rng(std::random_device{}());
  std::shuffle(h_x.begin(), h_x.end(), rng);
  std::shuffle(h_y.begin(), h_y.end(), rng);
  thrust::device_vector<float> d_mean_x(h_x.begin(), h_x.begin() + k);
  thrust::device_vector<float> d_mean_y(h_y.begin(), h_y.begin() + k);

  thrust::device_vector<float> d_sums_x(k);
  thrust::device_vector<float> d_sums_y(k);
  thrust::device_vector<int> d_counts(k, 0);

  const int threads = 1024;
  const int blocks = (number_of_elements + threads - 1) / threads;

  const auto start = std::chrono::high_resolution_clock::now();
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    thrust::fill(d_sums_x.begin(), d_sums_x.end(), 0);
    thrust::fill(d_sums_y.begin(), d_sums_y.end(), 0);
    thrust::fill(d_counts.begin(), d_counts.end(), 0);

    assign_clusters<<<blocks, threads>>>(d_x.data(),
                                         d_y.data(),
                                         number_of_elements,
                                         d_mean_x.data(),
                                         d_mean_y.data(),
                                         d_sums_x.data(),
                                         d_sums_y.data(),
                                         k,
                                         d_counts.data());
    cudaDeviceSynchronize();

    compute_new_means<<<1, k>>>(d_mean_x.data(),
                                d_mean_y.data(),
                                d_sums_x.data(),
                                d_sums_y.data(),
                                d_counts.data());
    cudaDeviceSynchronize();
  }
  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
  std::cerr << "Took: " << duration.count() << "s" << std::endl;

  thrust::host_vector<float> h_mean_x = d_mean_x;
  thrust::host_vector<float> h_mean_y = d_mean_y;

  for (size_t cluster = 0; cluster < k; ++cluster) {
    std::cout << h_mean_x[cluster] << " " << h_mean_y[cluster] << std::endl;
  }
}
