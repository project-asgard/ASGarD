#include "kronmult_cuda.hpp"
#include "build_info.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#define USE_GPU
#define GLOBAL_FUNCTION __global__
#define SYNCTHREADS __syncthreads()
#define SHARED_MEMORY __shared__
#define DEVICE_FUNCTION __device__
#define HOST_FUNCTION __host__
#else
#define GLOBAL_FUNCTION
#define SYNCTHREADS
#define SHARED_MEMORY
#define DEVICE_FUNCTION
#define HOST_FUNCTION
#endif

#include "kronmult1_xbatched.hpp"
#include "kronmult2_xbatched.hpp"
#include "kronmult3_xbatched.hpp"
#include "kronmult4_xbatched.hpp"
#include "kronmult5_xbatched.hpp"
#include "kronmult6_xbatched.hpp"

#ifdef ASGARD_USE_OPENMP
#include <omp.h>
#endif

// duplicated code from tools component - need host/device assert compiled
// separately
HOST_FUNCTION DEVICE_FUNCTION inline void expect(bool const condition)
{
  auto const ignore = [](auto ignored) { (void)ignored; };
  ignore(condition);
  assert(condition);
}

template<typename P>
GLOBAL_FUNCTION void
stage_inputs_kronmult_kernel(P const *const x, P *const workspace,
                             int const num_elems, int const num_copies)
{
#ifdef ASGARD_USE_CUDA

  expect(blockIdx.y == 0);
  expect(blockIdx.z == 0);
  expect(gridDim.y == 1);
  expect(gridDim.z == 1);

  auto const id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const num_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto const start       = id;
  auto const increment   = num_threads;
  auto const stop        = num_elems * num_copies;

  for (int64_t i = start; i < stop; i += increment)
  {
    workspace[i] = x[i % num_elems];
  }

#else

  auto const start     = 0;
  auto const increment = 1;
  auto const stop      = num_copies;

#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
  for (auto i = start; i < stop; i += increment)
  {
    auto const dest = workspace + i * num_elems;
    std::copy(x, x + num_elems, dest);
  }

#endif
}

template<typename P>
void stage_inputs_kronmult(P const *const x, P *const workspace,
                           int const num_elems, int const num_copies)
{
  expect(num_elems > 0);
  expect(num_copies > 0);

#ifdef ASGARD_USE_CUDA

  auto constexpr warp_size   = 32;
  auto constexpr num_warps   = 8;
  auto constexpr num_threads = num_warps * warp_size;

  auto const total_copies = static_cast<int64_t>(num_elems) * num_copies;
  auto const num_blocks   = (total_copies + num_threads - 1) / num_threads;

  stage_inputs_kronmult_kernel<P>
      <<<num_blocks, num_threads>>>(x, workspace, num_elems, num_copies);

  auto const stat = cudaDeviceSynchronize();
  expect(stat == cudaSuccess);
#else
  stage_inputs_kronmult_kernel(x, workspace, num_elems, num_copies);
#endif
}

// helper - given a cell and level coordinate, return a 1-dimensional index
DEVICE_FUNCTION
inline int get_1d_index(int const level, int const cell)

{
  expect((level >= 0) && (level < 30));
  expect(cell >= 0);

  if (level == 0)
  {
    return 0;
  }

  static int constexpr powers_of_2[] = {
      1,         2,        4,         8,         16,

      32,        64,       128,       256,       512,

      1024,      2048,     4096,      8192,      16384,

      32768,     65536,    131072,    262144,    524288,

      1048576,   2097152,  4194304,   8388608,   16777216,

      33554432,  67108864, 134217728, 268435456, 536870912,

      1073741824};

  return powers_of_2[level - 1] + cell;
}

// helper - calculate element coordinates -> operator matrix indices
DEVICE_FUNCTION
void get_indices(int const *const coords, int indices[], int const degree,
                 int const num_dims)
{
  expect(degree > 0);

  for (int i = 0; i < num_dims; ++i)
  {
    indices[i] = get_1d_index(coords[i], coords[i + num_dims]) * degree;
  }
}

template<typename P>
GLOBAL_FUNCTION void
prepare_kronmult_kernel(int const *const flattened_table,
                        P *const *const operators, int const operator_lda,
                        P *const element_x, P *const element_work, P *const fx,
                        P **const operator_ptrs, P **const work_ptrs,
                        P **const input_ptrs, P **const output_ptrs,
                        int const degree, int const num_terms,
                        int const num_dims, int const elem_row_start,
                        int const elem_row_stop, int const elem_col_start,
                        int const elem_col_stop)
{
  auto const num_cols = elem_col_stop - elem_col_start + 1;
  auto const num_rows = elem_row_stop - elem_row_start + 1;

  auto const deg_to_dim =
      static_cast<int>(pow((float)degree, (float)num_dims) + 0.1);

  auto const x_size     = static_cast<int64_t>(num_cols) * deg_to_dim;
  auto const coord_size = num_dims * 2;
  auto const num_elems  = static_cast<int64_t>(num_cols) * num_rows;

#ifdef ASGARD_USE_CUDA

  expect(blockIdx.y == 0);
  expect(blockIdx.z == 0);
  expect(gridDim.y == 1);
  expect(gridDim.z == 1);

  auto const id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const num_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;
  auto const start       = id;
  auto const increment   = num_threads;

#else
  auto const start     = 0;
  auto const increment = 1;
#endif

#ifndef ASGARD_USE_CUDA
#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
#endif
  for (int64_t i = start; i < num_elems; i += increment)
  {
    auto const row = i / num_cols + elem_row_start;
    auto const col = i % num_cols + elem_col_start;

    int constexpr max_dims = 6;
    expect(num_dims <= max_dims);

    // calculate and store operator row indices for this element
    int operator_row[max_dims];
    int const *const row_coords = flattened_table + coord_size * row;
    get_indices(row_coords, operator_row, degree, num_dims);

    // calculate and store operator col indices for this element
    int operator_col[max_dims];
    int const *const col_coords = flattened_table + coord_size * col;
    get_indices(col_coords, operator_col, degree, num_dims);

    auto const x_start =
        element_x + ((row - elem_row_start) * num_terms * x_size +
                     (col - elem_col_start) * deg_to_dim);

    for (auto t = 0; t < num_terms; ++t)
    {
      // get preallocated vector position for this kronmult
      auto const num_kron = (row - elem_row_start) * num_cols * num_terms +
                            (col - elem_col_start) * num_terms + t;

      // point to inputs
      input_ptrs[num_kron] = x_start + t * x_size;

      // point to work/output
      work_ptrs[num_kron]   = element_work + num_kron * deg_to_dim;
      output_ptrs[num_kron] = fx + (row - elem_row_start) * deg_to_dim;

      // point to operators
      auto const operator_start = num_kron * num_dims;
      for (auto d = 0; d < num_dims; ++d)
      {
        P *const coeff = operators[t * num_dims + d];
        operator_ptrs[operator_start + d] =
            coeff + operator_row[d] + operator_col[d] * operator_lda;
      }
    }
  }
}

// build batch lists for kronmult from simple
// arrays. built on device if cuda-enabled.
template<typename P>
void prepare_kronmult(int const *const flattened_table,
                      P *const *const operators, int const operator_lda,
                      P *const element_x, P *const element_work, P *const fx,
                      P **const operator_ptrs, P **const work_ptrs,
                      P **const input_ptrs, P **const output_ptrs,
                      int const degree, int const num_terms, int const num_dims,
                      int const elem_row_start, int const elem_row_stop,
                      int const elem_col_start, int const elem_col_stop)
{
  expect(elem_col_stop >= elem_col_start);
  expect(elem_row_stop >= elem_row_start);
  expect(elem_row_start >= 0);
  expect(elem_row_stop >= 0);
  expect(degree > 0);
  expect(num_terms > 0);
  expect(num_dims > 0);
  expect(flattened_table);
  expect(operators);
  expect(operator_lda > 0);
  expect(element_x);
  expect(element_work);
  expect(operator_ptrs);
  expect(work_ptrs);
  expect(input_ptrs);
  expect(output_ptrs);

#ifdef ASGARD_USE_CUDA
  auto constexpr warp_size   = 32;
  auto constexpr num_warps   = 8;
  auto constexpr num_threads = num_warps * warp_size;
  auto const num_krons =
      static_cast<int64_t>(elem_col_stop - elem_col_start + 1) *
      (elem_row_stop - elem_row_start + 1);
  auto const num_blocks = (num_krons / num_threads) + 1;
  prepare_kronmult_kernel<P><<<num_blocks, num_threads>>>(
      flattened_table, operators, operator_lda, element_x, element_work, fx,
      operator_ptrs, work_ptrs, input_ptrs, output_ptrs, degree, num_terms,
      num_dims, elem_row_start, elem_row_stop, elem_col_start, elem_col_stop);
  auto const stat = cudaDeviceSynchronize();
  expect(stat == cudaSuccess);
#else
  prepare_kronmult_kernel(
      flattened_table, operators, operator_lda, element_x, element_work, fx,
      operator_ptrs, work_ptrs, input_ptrs, output_ptrs, degree, num_terms,
      num_dims, elem_row_start, elem_row_stop, elem_col_start, elem_col_stop);
#endif
}

// call kronmult as function or kernel invocation
// --------------------------------------------
// note  the input memory referenced by x_ptrs will be over-written
// --------------------------------------------
template<typename P>
void call_kronmult(int const n, P *x_ptrs[], P *output_ptrs[], P *work_ptrs[],
                   P const *const operator_ptrs[], int const lda,
                   int const num_krons, int const num_dims)
{
#ifdef ASGARD_USE_CUDA
  {
    int constexpr warpsize    = 32;
    int constexpr nwarps      = 1;
    int constexpr num_threads = nwarps * warpsize;

    switch (num_dims)
    {
    case 1:
      kronmult1_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 2:
      kronmult2_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 3:
      kronmult3_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 4:
      kronmult4_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 5:
      kronmult5_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    case 6:
      kronmult6_xbatched<P><<<num_krons, num_threads>>>(
          n, operator_ptrs, lda, x_ptrs, output_ptrs, work_ptrs, num_krons);
      break;
    default:
      expect(false);
    };

    // -------------------------------------------
    // note important to wait for kernel to finish
    // -------------------------------------------
    auto const stat = cudaDeviceSynchronize();
    expect(stat == cudaSuccess);
  }
#else

  {
    switch (num_dims)
    {
    case 1:
      kronmult1_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 2:
      kronmult2_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 3:
      kronmult3_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 4:
      kronmult4_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 5:
      kronmult5_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    case 6:
      kronmult6_xbatched<P>(n, operator_ptrs, lda, x_ptrs, output_ptrs,
                            work_ptrs, num_krons);
      break;
    default:
      expect(false);
    };
  }
#endif
}

#define X(T)                                                                \
  template void stage_inputs_kronmult(T const *const x, T *const workspace, \
                                      int const num_elems,                  \
                                      int const num_copies);
#include "../type_list_float.inc"
#undef X

#define X(T)                                                             \
  template void prepare_kronmult(                                        \
      int const *const flattened_table, T *const *const operators,       \
      int const operator_lda, T *const element_x, T *const element_work, \
      T *const fx, T **const operator_ptrs, T **const work_ptrs,         \
      T **const input_ptrs, T **const output_ptrs, int const degree,     \
      int const num_terms, int const num_dims, int const elem_row_start, \
      int const elem_row_stop, int const elem_col_start,                 \
      int const elem_col_stop);
#include "../type_list_float.inc"
#undef X

#define X(T)                                                                  \
  template void call_kronmult(int const n, T *x_ptrs[], T *output_ptrs[],     \
                              T *work_ptrs[], T const *const operator_ptrs[], \
                              int const lda, int const num_krons,             \
                              int const num_dims);
#include "../type_list_float.inc"
#undef X
