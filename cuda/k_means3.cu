#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

struct Data {
  Data(int size) : size(size), bytes(size * sizeof(float)) {
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemset(x, 0, bytes);
    cudaMemset(y, 0, bytes);
  }

  Data(int size, std::vector<float>& h_x, std::vector<float>& h_y)
  : size(size), bytes(size * sizeof(float)) {
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMemcpy(x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, h_y.data(), bytes, cudaMemcpyHostToDevice);
  }

  ~Data() {
    cudaFree(x);
    cudaFree(y);
  }

  float* x{nullptr};
  float* y{nullptr};
  int size{0};
  int bytes{0};
};

__device__ float
squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__global__ void fine_reduce(const float* __restrict__ data_x,
                            const float* __restrict__ data_y,
                            int data_size,
                            const float* __restrict__ means_x,
                            const float* __restrict__ means_y,
                            float* __restrict__ new_sums_x,
                            float* __restrict__ new_sums_y,
                            int k,
                            int* __restrict__ counts) {
  extern __shared__ float shared_data[];

  const int local_index = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= data_size) return;

  // Load the mean values into shared memory.
  if (local_index < k) {
    shared_data[local_index] = means_x[local_index];
    shared_data[k + local_index] = means_y[local_index];
  }

  __syncthreads();

  // Load once here.
  const float x_value = data_x[global_index];
  const float y_value = data_y[global_index];

  float best_distance = FLT_MAX;
  int best_cluster = -1;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance = squared_l2_distance(x_value,
                                               y_value,
                                               shared_data[cluster],
                                               shared_data[k + cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  __syncthreads();

  // reduction

  const int x = local_index;
  const int y = local_index + blockDim.x;
  const int count = local_index + blockDim.x + blockDim.x;

  for (int cluster = 0; cluster < k; ++cluster) {
    shared_data[x] = (best_cluster == cluster) ? x_value : 0;
    shared_data[y] = (best_cluster == cluster) ? y_value : 0;
    shared_data[count] = (best_cluster == cluster) ? 1 : 0;
    __syncthreads();

    // Reduction for this cluster.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (local_index < stride) {
        shared_data[x] += shared_data[x + stride];
        shared_data[y] += shared_data[y + stride];
        shared_data[count] += shared_data[count + stride];
      }
      __syncthreads();
    }

    // Now shared_data[0] holds the sum for x.

    if (local_index == 0) {
      const int cluster_index = blockIdx.x * k + cluster;
      new_sums_x[cluster_index] = shared_data[x];
      new_sums_y[cluster_index] = shared_data[y];
      counts[cluster_index] = shared_data[count];
    }
    __syncthreads();
  }
}

__global__ void coarse_reduce(float* __restrict__ means_x,
                              float* __restrict__ means_y,
                              float* __restrict__ new_sum_x,
                              float* __restrict__ new_sum_y,
                              int k,
                              int* __restrict__ counts) {
  extern __shared__ float shared_data[];

  const int index = threadIdx.x;
  const int y_offset = blockDim.x;

  shared_data[index] = new_sum_x[index];
  shared_data[y_offset + index] = new_sum_y[index];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= k; stride /= 2) {
    if (index < stride) {
      shared_data[index] += shared_data[index + stride];
      shared_data[y_offset + index] += shared_data[y_offset + index + stride];
    }
    __syncthreads();
  }

  if (index < k) {
    const int count = max(1, counts[index]);
    means_x[index] = new_sum_x[index] / count;
    means_y[index] = new_sum_y[index] / count;
    new_sum_y[index] = 0;
    new_sum_x[index] = 0;
    counts[index] = 0;
  }
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: k-means <data-file> <k> [iterations]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto k = std::atoi(argv[2]);
  const auto number_of_iterations = (argc == 4) ? std::atoi(argv[3]) : 300;

  std::vector<float> h_x;
  std::vector<float> h_y;
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

  Data d_data(number_of_elements, h_x, h_y);

  std::mt19937 rng(std::random_device{}());
  std::shuffle(h_x.begin(), h_x.end(), rng);
  std::shuffle(h_y.begin(), h_y.end(), rng);
  Data d_means(k, h_x, h_y);

  const int threads = 1024;
  const int blocks = (number_of_elements + threads - 1) / threads;

  std::cerr << "Processing " << number_of_elements << " points on " << blocks
            << " blocks x " << threads << " threads" << std::endl;

  // * 3 for x, y and counts.
  const int fine_shared_memory = 3 * threads * sizeof(float);
  // * 2 for x and y. Will have k * blocks threads for the coarse reduction.
  const int coarse_shared_memory = 2 * k * blocks * sizeof(float);

  Data d_sums(k * blocks);
  int* d_counts;
  cudaMalloc(&d_counts, k * blocks * sizeof(int));
  cudaMemset(d_counts, 0, k * blocks * sizeof(int));

  const auto start = std::chrono::high_resolution_clock::now();
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    fine_reduce<<<blocks, threads, fine_shared_memory>>>(d_data.x,
                                                         d_data.y,
                                                         d_data.size,
                                                         d_means.x,
                                                         d_means.y,
                                                         d_sums.x,
                                                         d_sums.y,
                                                         k,
                                                         d_counts);
    cudaDeviceSynchronize();

    coarse_reduce<<<1, k * blocks, coarse_shared_memory>>>(d_means.x,
                                                           d_means.y,
                                                           d_sums.x,
                                                           d_sums.y,
                                                           k,
                                                           d_counts);

    cudaDeviceSynchronize();
  }
  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
  std::cerr << "Took: " << duration.count() << "s" << std::endl;

  cudaFree(d_counts);

  std::vector<float> mean_x(k, 0);
  std::vector<float> mean_y(k, 0);
  cudaMemcpy(mean_x.data(), d_means.x, d_means.bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(mean_y.data(), d_means.y, d_means.bytes, cudaMemcpyDeviceToHost);

  for (size_t cluster = 0; cluster < k; ++cluster) {
    std::cout << mean_x[cluster] << " " << mean_y[cluster] << std::endl;
  }
}
