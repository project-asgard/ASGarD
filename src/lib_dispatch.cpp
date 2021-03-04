#include "lib_dispatch.hpp"
#include "build_info.hpp"
#include "tools.hpp"

#include <cmath>
#include <iostream>
#include <type_traits>

#ifdef ASGARD_USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#ifdef ASGARD_USE_SLATE
extern "C" void slate_sgesv_(const int *n, const int *nrhs, float *a,
                             const int *lda, int *ipiv, float *b,
                             const int *ldb, int *info);

extern "C" void slate_dgesv_(const int *n, const int *nrhs, double *a,
                             const int *lda, int *ipiv, double *b,
                             const int *ldb, int *info);

extern "C" void slate_dgetrs_(const char *trans, const int *n, const int *nrhs,
                              double *A, const int *lda, int *ipiv, double *b,
                              const int *ldb, int *info);

extern "C" void slate_sgetrs_(const char *trans, const int *n, const int *nrhs,
                              float *A, const int *lda, int *ipiv, float *b,
                              const int *ldb, int *info);
#endif

auto const ignore = [](auto ignored) { (void)ignored; };
struct device_handler
{
  device_handler()
  {
#ifdef ASGARD_USE_CUDA
    auto success = cublasCreate(&handle);
    expect(success == CUBLAS_STATUS_SUCCESS);

    success = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    expect(success == CUBLAS_STATUS_SUCCESS);
#endif
  }

  void set_device(int const local_rank)
  {
#ifdef ASGARD_USE_CUDA
    int num_devices;
    auto success = cudaGetDeviceCount(&num_devices);

    expect(success == cudaSuccess);
    expect(local_rank >= 0);
    expect(local_rank < num_devices);

    if (handle)
    {
      auto const cublas_success = cublasDestroy(handle);
      expect(cublas_success == CUBLAS_STATUS_SUCCESS);
    }

    success = cudaSetDevice(local_rank);
    expect(success == cudaSuccess);
    auto const cublas_success = cublasCreate(&handle);
    expect(cublas_success == CUBLAS_STATUS_SUCCESS);

#else
    ignore(local_rank);
#endif
  }

#ifdef ASGARD_USE_CUDA
  cublasHandle_t const &get_handle() const { return handle; }
#endif
  ~device_handler()
  {
#ifdef ASGARD_USE_CUDA
    cublasDestroy(handle);
#endif
  }

private:
#ifdef ASGARD_USE_CUDA
  cublasHandle_t handle;
#endif
};
static device_handler device;

void initialize_libraries(int const local_rank)
{
#ifdef ASGARD_USE_CUDA
  expect(local_rank >= 0);
  device.set_device(local_rank);
#else
  ignore(local_rank);
#endif
}

#ifdef ASGARD_USE_CUDA
inline cublasOperation_t cublas_trans(char trans)
{
  if (trans == 'N' || trans == 'n')
  {
    return CUBLAS_OP_N;
  }
  else
  {
    return CUBLAS_OP_T;
  }
}
#endif

namespace lib_dispatch
{
template<typename P>
void rotg(P *a, P *b, P *c, P *s, resource const resrc)
{
  expect(a && b && c && s);

  // function doesn't make sense outside of FP context
  expect(std::is_floating_point_v<P>);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // function instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDrotg(device.get_handle(), a, b, c, s);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSrotg(device.get_handle(), a, b, c, s);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    drotg_(a, b, c, s);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    srotg_(a, b, c, s);
  }
}

template<typename P>
P nrm2(int *n, P *x, int *incx, resource const resrc)
{
  expect(x);
  expect(incx && *incx >= 0);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);
    P norm;
    // function instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDnrm2(device.get_handle(), *n, x, *incx, &norm);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSnrm2(device.get_handle(), *n, x, *incx, &norm);
      expect(success == 0);
    }
    return norm;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    return dnrm2_(n, x, incx);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    return snrm2_(n, x, incx);
  }
  else
  {
    auto sum_square = 0.0;
    for (int i = 0; i < *n; ++i)
    {
      sum_square += std::pow(x[i * (*incx)], 2);
    }
    return std::sqrt(sum_square);
  }
}

template<typename P>
void copy(int *n, P *x, int *incx, P *y, int *incy, resource const resrc)
{
  expect(x);
  expect(y);
  expect(incx && *incx >= 0);
  expect(incy && *incy >= 0);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // function instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDcopy(device.get_handle(), *n, x, *incx, y, *incy);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasScopy(device.get_handle(), *n, x, *incx, y, *incy);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    dcopy_(n, x, incx, y, incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    scopy_(n, x, incx, y, incy);
  }
  else
  {
    for (int i = 0; i < *n; ++i)
    {
      y[i * (*incy)] = x[i * (*incx)];
    }
  }
}

template<typename P>
P dot(int *n, P *x, int *incx, P *y, int *incy, resource const resrc)
{
  expect(x);
  expect(y);
  expect(incx && *incx >= 0);
  expect(incy && *incy >= 0);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    P result = 0.;
    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDdot(device.get_handle(), *n, x, *incx, y, *incy, &result);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSdot(device.get_handle(), *n, x, *incx, y, *incy, &result);
      expect(success == 0);
    }
    return result;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    return ddot_(n, x, incx, y, incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    return sdot_(n, x, incx, y, incy);
  }
  else
  {
    P ans = 0.0;
    for (int i = 0; i < *n; ++i)
    {
      ans += x[i * (*incx)] * y[i * (*incy)];
    }
    return ans;
  }
}

template<typename P>
void axpy(int *n, P *alpha, P *x, int *incx, P *y, int *incy,
          resource const resrc)
{
  expect(alpha);
  expect(x);
  expect(y);
  expect(incx && *incx >= 0);
  expect(incy && *incy >= 0);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDaxpy(device.get_handle(), *n, alpha, x, *incx, y, *incy);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSaxpy(device.get_handle(), *n, alpha, x, *incx, y, *incy);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    daxpy_(n, alpha, x, incx, y, incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    saxpy_(n, alpha, x, incx, y, incy);
  }
  else
  {
    for (int i = 0; i < *n; ++i)
    {
      y[i * (*incy)] = y[i * (*incy)] + x[i * (*incx)] * (*alpha);
    }
  }
}

template<typename P>
void scal(int *n, P *alpha, P *x, int *incx, resource const resrc)
{
  expect(alpha);
  expect(x);
  expect(n && *n >= 0);
  expect(incx && *incx >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDscal(device.get_handle(), *n, alpha, x, *incx);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSscal(device.get_handle(), *n, alpha, x, *incx);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    dscal_(n, alpha, x, incx);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sscal_(n, alpha, x, incx);
  }
  else
  {
    for (int i = 0; i < *n; ++i)
    {
      x[i * (*incx)] *= *alpha;
    }
  }
}

//
// Simple helpers for non-float types
//
template<typename P>
static void
basic_gemm(P const *A, bool const trans_A, int const lda, P *B, bool trans_B,
           int const ldb, P *C, int const ldc, int const m, int const k,
           int const n, P const alpha, P const beta)
{
  expect(m > 0);
  expect(k > 0);
  expect(n > 0);
  expect(lda > 0); // FIXME Tyler says these could be more thorough
  expect(ldb > 0);
  expect(ldc > 0);

  for (auto i = 0; i < m; ++i)
  {
    for (auto j = 0; j < n; ++j)
    {
      P result = 0.0;
      for (auto z = 0; z < k; ++z)
      {
        int const A_loc = trans_A ? i * lda + z : z * lda + i;
        int const B_loc = trans_B ? z * ldb + j : j * ldb + z;
        result += A[A_loc] * B[B_loc];
      }
      C[j * ldc + i] = C[j * ldc + i] * beta + alpha * result;
    }
  }
}

template<typename P>
static void basic_gemv(P const *A, bool const trans_A, int const lda,
                       P const *x, int const incx, P *y, int const incy,
                       int const m, int const n, P const alpha, P const beta)
{
  expect(m > 0);
  expect(n > 0);
  expect(lda > 0);
  expect(incx > 0);
  expect(incy > 0);

  for (auto i = 0; i < m; ++i)
  {
    P result = 0.0;
    for (auto j = 0; j < n; ++j)
    {
      int const A_loc = trans_A ? i * lda + j : j * lda + i;
      result += A[A_loc] * x[j * incx];
    }
    y[i * incy] = y[i * incy] * beta + alpha * result;
  }
}

template<typename P>
void gemv(char const *trans, int *m, int *n, P *alpha, P *A, int *lda, P *x,
          int *incx, P *beta, P *y, int *incy, resource const resrc)
{
  expect(alpha);
  expect(A);
  expect(x);
  expect(beta);
  expect(y);
  expect(m && *m >= 0);
  expect(n && *n >= 0);
  expect(lda && *lda >= 0);
  expect(incx && *incx >= 0);
  expect(incy && *incy >= 0);
  expect(trans && (*trans == 't' || *trans == 'n'));

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success =
          cublasDgemv(device.get_handle(), cublas_trans(*trans), *m, *n, alpha,
                      A, *lda, x, *incx, beta, y, *incy);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success =
          cublasSgemv(device.get_handle(), cublas_trans(*trans), *m, *n, alpha,
                      A, *lda, x, *incx, beta, y, *incy);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    dgemv_(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgemv_(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  }
  else
  {
    bool const trans_A = (*trans == 't') ? true : false;
    int const rows_A   = trans_A ? *n : *m;
    int const cols_A   = trans_A ? *m : *n;
    basic_gemv(A, trans_A, *lda, x, *incx, y, *incy, rows_A, cols_A, *alpha,
               *beta);
  }
}

template<typename P>
void gemm(char const *transa, char const *transb, int *m, int *n, int *k,
          P *alpha, P *A, int *lda, P *B, int *ldb, P *beta, P *C, int *ldc,
          resource const resrc)
{
  expect(alpha);
  expect(A);
  expect(lda && *lda >= 0);
  expect(B);
  expect(ldb && *ldb >= 0);
  expect(beta);
  expect(C);
  expect(ldc && *ldc >= 0);
  expect(m && *m >= 0);
  expect(n && *n >= 0);
  expect(k && *k >= 0);
  expect(transa && (*transa == 't' || *transa == 'n'));
  expect(transb && (*transb == 't' || *transb == 'n'));

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgemm(
          device.get_handle(), cublas_trans(*transa), cublas_trans(*transb), *m,
          *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgemm(
          device.get_handle(), cublas_trans(*transa), cublas_trans(*transb), *m,
          *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    dgemm_(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgemm_(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  else
  {
    bool const trans_A = (*transa == 't') ? true : false;
    bool const trans_B = (*transb == 't') ? true : false;
    basic_gemm(A, trans_A, *lda, B, trans_B, *ldb, C, *ldc, *m, *k, *n, *alpha,
               *beta);
  }
}

template<typename P>
void getrf(int *m, int *n, P *A, int *lda, int *ipiv, int *info,
           resource const resrc)
{
  expect(A);
  expect(ipiv);
  expect(info);
  expect(lda && *lda >= 0);
  expect(m && *m >= 0);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA

    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);
    expect(*m == *n);
    ignore(m);

    P **A_d;
    auto stat = cudaMalloc((void **)&A_d, sizeof(P *));
    expect(stat == 0);
    stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgetrfBatched(device.get_handle(), *n, A_d,
                                               *lda, ipiv, info, 1);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgetrfBatched(device.get_handle(), *n, A_d,
                                               *lda, ipiv, info, 1);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    dgetrf_(m, n, A, lda, ipiv, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgetrf_(m, n, A, lda, ipiv, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getrf not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void getri(int *n, P *A, int *lda, int *ipiv, P *work, int *lwork, int *info,
           resource const resrc)
{
  expect(A);
  expect(ipiv);
  expect(work);
  expect(lwork);
  expect(info);
  expect(lda && *lda >= 0);
  expect(n && *n >= 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA

    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    expect(*lwork == (*n) * (*n));
    ignore(lwork);

    P const **A_d;
    P **work_d;
    auto stat = cudaMalloc((void **)&A_d, sizeof(P *));
    expect(stat == 0);
    stat = cudaMalloc((void **)&work_d, sizeof(P *));
    expect(stat == 0);

    stat = cudaMemcpy(A_d, &A, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == 0);
    stat = cudaMemcpy(work_d, &work, sizeof(P *), cudaMemcpyHostToDevice);
    expect(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgetriBatched(
          device.get_handle(), *n, A_d, *lda, nullptr, work_d, *n, info, 1);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgetriBatched(
          device.get_handle(), *n, A_d, *lda, nullptr, work_d, *n, info, 1);
      expect(success == 0);
    }
    return;
#endif
  }

  // default execution on the host for any resource
  if constexpr (std::is_same<P, double>::value)
  {
    dgetri_(n, A, lda, ipiv, work, lwork, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgetri_(n, A, lda, ipiv, work, lwork, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getri not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void batched_gemm(P **const &a, int *lda, char const *transa, P **const &b,
                  int *ldb, char const *transb, P **const &c, int *ldc, int *m,
                  int *n, int *k, P *alpha, P *beta, int *num_batch,
                  resource const resrc)
{
  expect(alpha);
  expect(a);
  expect(lda && *lda >= 0);
  expect(b);
  expect(ldb && *ldb >= 0);
  expect(beta);
  expect(c);
  expect(ldc && *ldc >= 0);
  expect(m && *m >= 0);
  expect(n && *n >= 0);
  expect(k && *k >= 0);
  expect(transa && (*transa == 't' || *transa == 'n'));
  expect(transb && (*transb == 't' || *transb == 'n'));
  expect(num_batch && *num_batch > 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);

    P const **a_d;
    P const **b_d;
    P **c_d;
    size_t const list_size = *num_batch * sizeof(P *);

    auto stat = cudaMalloc((void **)&a_d, list_size);
    expect(stat == 0);
    stat = cudaMalloc((void **)&b_d, list_size);
    expect(stat == 0);
    stat = cudaMalloc((void **)&c_d, list_size);
    expect(stat == 0);
    stat = cudaMemcpy(a_d, a, list_size, cudaMemcpyHostToDevice);
    expect(stat == 0);
    stat = cudaMemcpy(b_d, b, list_size, cudaMemcpyHostToDevice);
    expect(stat == 0);
    stat = cudaMemcpy(c_d, c, list_size, cudaMemcpyHostToDevice);
    expect(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgemmBatched(
          device.get_handle(), cublas_trans(*transa), cublas_trans(*transb), *m,
          *n, *k, alpha, a_d, *lda, b_d, *ldb, beta, c_d, *ldc, *num_batch);
      auto const cuda_stat = cudaDeviceSynchronize();
      expect(cuda_stat == 0);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgemmBatched(
          device.get_handle(), cublas_trans(*transa), cublas_trans(*transb), *m,
          *n, *k, alpha, a_d, *lda, b_d, *ldb, beta, c_d, *ldc, *num_batch);
      auto const cuda_stat = cudaDeviceSynchronize();
      expect(cuda_stat == 0);
      expect(success == 0);
    }

    stat = cudaFree(a_d);
    expect(stat == 0);
    stat = cudaFree(b_d);
    expect(stat == 0);
    stat = cudaFree(c_d);
    expect(stat == 0);

    return;
#endif
  }

  // default execution on the host for any resource
  int const end = *num_batch;
#ifdef ASGARD_USE_OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < end; ++i)
  {
    gemm(transa, transb, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc,
         resource::host);
  }
}

// resrctricted subset of gemv functionality provided by
// calling batched gemm - no non-unit increments allowed for
// x or y for now
template<typename P>
void batched_gemv(P **const &a, int *lda, char const *trans, P **const &x,
                  P **const &y, int *m, int *n, P *alpha, P *beta,
                  int *num_batch, resource const resrc)
{
  expect(alpha);
  expect(a);
  expect(lda && *lda >= 0);
  expect(x);
  expect(beta);
  expect(y);
  expect(m && *m >= 0);
  expect(n && *n >= 0);

  expect(trans && (*trans == 't' || *trans == 'n'));
  expect(num_batch && *num_batch > 0);

  if (resrc == resource::device)
  {
    // device-specific specialization if needed
#ifdef ASGARD_USE_CUDA
    // no non-fp blas on device
    expect(std::is_floating_point_v<P>);
    char const transb = 'n';

    int gemm_m = *trans == 't' ? *n : *m;
    int gemm_k = *trans == 't' ? *m : *n;
    int gemm_n = 1;

    int ldb = gemm_k;
    int ldc = gemm_m;

    P const **a_d;
    P const **x_d;
    P **y_d;
    size_t const list_size = *num_batch * sizeof(P *);

    auto stat = cudaMalloc((void **)&a_d, list_size);
    expect(stat == 0);
    stat = cudaMalloc((void **)&x_d, list_size);
    expect(stat == 0);
    stat = cudaMalloc((void **)&y_d, list_size);
    expect(stat == 0);
    stat = cudaMemcpy(a_d, a, list_size, cudaMemcpyHostToDevice);
    expect(stat == 0);
    stat = cudaMemcpy(x_d, x, list_size, cudaMemcpyHostToDevice);
    expect(stat == 0);
    stat = cudaMemcpy(y_d, y, list_size, cudaMemcpyHostToDevice);
    expect(stat == 0);

    // instantiated for these two fp types
    if constexpr (std::is_same<P, double>::value)
    {
      auto const success = cublasDgemmBatched(
          device.get_handle(), cublas_trans(*trans), cublas_trans(transb),
          gemm_m, gemm_n, gemm_k, alpha, a_d, *lda, x_d, ldb, beta, y_d, ldc,
          *num_batch);
      auto const cuda_stat = cudaDeviceSynchronize();
      expect(cuda_stat == 0);
      expect(success == 0);
    }
    else if constexpr (std::is_same<P, float>::value)
    {
      auto const success = cublasSgemmBatched(
          device.get_handle(), cublas_trans(*trans), cublas_trans(transb),
          gemm_m, gemm_n, gemm_k, alpha, a_d, *lda, x_d, ldb, beta, y_d, ldc,
          *num_batch);
      auto const cuda_stat = cudaDeviceSynchronize();
      expect(cuda_stat == 0);
      expect(success == 0);
    }

    stat = cudaFree(a_d);
    expect(stat == 0);
    stat = cudaFree(x_d);
    expect(stat == 0);
    stat = cudaFree(y_d);
    expect(stat == 0);

    return;

#endif
  }

  // default execution on the host for any resource
  int incx = 1;
  int incy = 1;
  for (int i = 0; i < *num_batch; ++i)
  {
    gemv(trans, m, n, alpha, a[i], lda, x[i], &incx, beta, y[i], &incy,
         resource::host);
  }
}

template<typename P>
void gesv(int *n, int *nrhs, P *A, int *lda, int *ipiv, P *b, int *ldb,
          int *info)
{
  expect(n);
  expect(nrhs);
  expect(A);
  expect(lda);
  expect(ipiv);
  expect(info);
  expect(b);
  expect(ldb);
  expect(*ldb >= 1);
  expect(*lda >= 1);
  expect(*n >= 0);
  if constexpr (std::is_same<P, double>::value)
  {
    dgesv_(n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgesv_(n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "gesv not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void getrs(char *trans, int *n, int *nrhs, P *A, int *lda, int *ipiv, P *b,
           int *ldb, int *info)
{
  expect(trans);
  expect(n);
  expect(nrhs);
  expect(A);
  expect(lda);
  expect(ipiv);
  expect(info);
  expect(b);
  expect(ldb);
  expect(*ldb >= 1);
  expect(*lda >= 1);
  expect(*n >= 0);
  if constexpr (std::is_same<P, double>::value)
  {
    dgetrs_(trans, n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    sgetrs_(trans, n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getrs not implemented for non-floating types" << '\n';
    assert(false);
  }
}

#ifdef ASGARD_USE_SLATE
template<typename P>
void slate_gesv(int *n, int *nrhs, P *A, int *lda, int *ipiv, P *b, int *ldb,
                int *info)
{
  expect(n);
  expect(nrhs);
  expect(A);
  expect(lda);
  expect(ipiv);
  expect(info);
  expect(b);
  expect(ldb);
  expect(*ldb >= 1);
  expect(*lda >= 1);
  expect(*n >= 0);
  if constexpr (std::is_same<P, double>::value)
  {
    slate_dgesv_(n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    slate_sgesv_(n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "gesv not implemented for non-floating types" << '\n';
    tools::expect(false);
  }
}

template<typename P>
void slate_getrs(char *trans, int *n, int *nrhs, P *A, int *lda, int *ipiv,
                 P *b, int *ldb, int *info)
{
  expect(trans);
  expect(n);
  expect(nrhs);
  expect(A);
  expect(lda);
  expect(ipiv);
  expect(info);
  expect(b);
  expect(ldb);
  expect(*ldb >= 1);
  expect(*lda >= 1);
  expect(*n >= 0);
  if constexpr (std::is_same<P, double>::value)
  {
    slate_dgetrs_(trans, n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    slate_sgetrs_(trans, n, nrhs, A, lda, ipiv, b, ldb, info);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "getrs not implemented for non-floating types" << '\n';
    expect(false);
  }
}
#endif

#define X(T) template void rotg(T *, T *, T *, T *, resource const resrc);
#include "type_list_float.inc"
#undef X

#define X(T) template T nrm2(int *n, T *x, int *incx, resource const resrc);
#include "type_list_float.inc"
#undef X

#define X(T)                                                   \
  template void copy(int *n, T *x, int *incx, T *y, int *incy, \
                     resource const resrc);
#include "type_list.inc"
#undef X

#define X(T)                                               \
  template T dot(int *n, T *x, int *incx, T *y, int *incy, \
                 resource const resrc);
#include "type_list.inc"
#undef X

#define X(T)                                                             \
  template void axpy(int *n, T *alpha, T *x, int *incx, T *y, int *incy, \
                     resource const resrc);
#include "type_list.inc"
#undef X

#define X(T) \
  template void scal(int *n, T *alpha, T *x, int *incx, resource const resrc);
#include "type_list.inc"
#undef X

#define X(T)                                                              \
  template void gemv(char const *trans, int *m, int *n, T *alpha, T *A,   \
                     int *lda, T *x, int *incx, T *beta, T *y, int *incy, \
                     resource const resrc);
#include "type_list.inc"
#undef X

#define X(T)                                                                 \
  template void gemm(char const *transa, char const *transb, int *m, int *n, \
                     int *k, T *alpha, T *A, int *lda, T *B, int *ldb,       \
                     T *beta, T *C, int *ldc, resource const resrc);
#include "type_list.inc"
#undef X

#define X(T)                                                                \
  template void getrf(int *m, int *n, T *A, int *lda, int *ipiv, int *info, \
                      resource const resrc);
#include "type_list_float.inc"
#undef X

#define X(T)                                                                  \
  template void getri(int *n, T *A, int *lda, int *ipiv, T *work, int *lwork, \
                      int *info, resource const resrc);
#include "type_list_float.inc"
#undef X

#define X(T)                                                              \
  template void batched_gemm(                                             \
      T **const &a, int *lda, char const *transa, T **const &b, int *ldb, \
      char const *transb, T **const &c, int *ldc, int *m, int *n, int *k, \
      T *alpha, T *beta, int *num_batch, resource const resrc);
#include "type_list_float.inc"
#undef X

#define X(T)                                                             \
  template void batched_gemv(T **const &a, int *lda, char const *transa, \
                             T **const &x, T **const &y, int *m, int *n, \
                             T *alpha, T *beta, int *num_batch,          \
                             resource const resrc);
#include "type_list_float.inc"
#undef X

#define X(T)                                                             \
  template void gesv(int *n, int *nrhs, T *A, int *lda, int *ipiv, T *b, \
                     int *ldb, int *info);
#include "type_list_float.inc"
#undef X

#define X(T)                                                          \
  template void getrs(char *trans, int *n, int *nrhs, T *A, int *lda, \
                      int *ipiv, T *b, int *ldb, int *info);
#include "type_list_float.inc"
#undef x

#ifdef ASGARD_USE_SLATE
#define X(T)                                                                   \
  template void slate_gesv(int *n, int *nrhs, T *A, int *lda, int *ipiv, T *b, \
                           int *ldb, int *info);
#include "type_list_float.inc"
#undef X

#define X(T)                                                                \
  template void slate_getrs(char *trans, int *n, int *nrhs, T *A, int *lda, \
                            int *ipiv, T *b, int *ldb, int *info);
#include "type_list_float.inc"
#undef X
#endif
} // namespace lib_dispatch
