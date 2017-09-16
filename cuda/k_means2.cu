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

__global__ void assign_clusters(const float* __restrict__ data_x,
                                const float* __restrict__ data_y,
                                int data_size,
                                const float* __restrict__ means_x,
                                const float* __restrict__ means_y,
                                float* __restrict__ new_sums_x,
                                float* __restrict__ new_sums_y,
                                int k,
                                int* __restrict__ counts) {
  extern __shared__ float shared_means[];

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= data_size) return;

  if (threadIdx.x < k) {
    shared_means[threadIdx.x] = means_x[threadIdx.x];
    shared_means[k + threadIdx.x] = means_y[threadIdx.x];
  }

  __syncthreads();

  // Make global loads once.
  const float x = data_x[index];
  const float y = data_y[index];

  float best_distance = FLT_MAX;
  int best_cluster = 0;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance = squared_l2_distance(x,
                                               y,
                                               shared_means[cluster],
                                               shared_means[k + cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  atomicAdd(&new_sums_x[best_cluster], x);
  atomicAdd(&new_sums_y[best_cluster], y);
  atomicAdd(&counts[best_cluster], 1);
}

__global__ void compute_new_means_and_reset(float* __restrict__ means_x,
                                            float* __restrict__ means_y,
                                            float* __restrict__ new_sum_x,
                                            float* __restrict__ new_sum_y,
                                            int* __restrict__ counts) {
  const int cluster = threadIdx.x;
  const int count = max(1, counts[cluster]);
  means_x[cluster] = new_sum_x[cluster] / count;
  means_y[cluster] = new_sum_y[cluster] / count;

  new_sum_y[cluster] = 0;
  new_sum_x[cluster] = 0;
  counts[cluster] = 0;
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: assign_clusters <data-file> <k> [iterations]"
              << std::endl;
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

  Data d_sums(k);

  int* d_counts;
  cudaMalloc(&d_counts, k * sizeof(int));
  cudaMemset(d_counts, 0, k * sizeof(int));

  const int threads = 1024;
  const int blocks = (number_of_elements + threads - 1) / threads;
  const int shared_memory = d_means.bytes * 2;

  std::cerr << "Processing " << number_of_elements << " points on " << blocks
            << " blocks x " << threads << " threads" << std::endl;

  const auto start = std::chrono::high_resolution_clock::now();
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    assign_clusters<<<blocks, threads, shared_memory>>>(d_data.x,
                                                        d_data.y,
                                                        d_data.size,
                                                        d_means.x,
                                                        d_means.y,
                                                        d_sums.x,
                                                        d_sums.y,
                                                        k,
                                                        d_counts);
    cudaDeviceSynchronize();

    compute_new_means_and_reset<<<1, k>>>(d_means.x,
                                          d_means.y,
                                          d_sums.x,
                                          d_sums.y,
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
