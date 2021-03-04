#include "transformations.hpp"
#include "batch.hpp"
#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "tools.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>

//
// set the range specified by first and last,
// starting with the supplied value and incrementing
// that value by stride for each position in the range.
//
template<typename ForwardIterator, typename P>
static void strided_iota(ForwardIterator first, ForwardIterator last, P value,
                         P const stride)
{
  while (first != last)
  {
    *first++ = value;
    value += stride;
  }
}

// perform recursive kronecker product
template<typename P>
fk::vector<P>
kron_d(std::vector<fk::vector<P>> const &operands, int const num_prods)
{
  expect(num_prods > 0);
  if (num_prods == 1)
  {
    return operands[0];
  }
  if (num_prods == 2)
  {
    return operands[0].single_column_kron(operands[1]);
  }
  return kron_d(operands, num_prods - 1)
      .single_column_kron(operands[num_prods - 1]);
}

/* calculate required mb for the matrix resulting from the kron prod of a
 * sequence of matrices */
template<typename P>
int kron_matrix_MB(
    std::vector<fk::matrix<P, mem_type::const_view>> const &kron_matrices)
{
  long r = 1;
  long c = 1;

  for (int i = 0; i < static_cast<int>(kron_matrices.size()); ++i)
  {
    r *= kron_matrices[i].nrows();
    c *= kron_matrices[i].ncols();
  }

  return r * c * sizeof(P) * 1e-6;
}

/* given a vector of matrices, return the Kronecker product of all of them in
 * order */
template<typename P>
fk::matrix<P>
recursive_kron(std::vector<fk::matrix<P, mem_type::view>> &kron_matrices,
               int const index)
{
  expect(index >= 0);
  expect(index < static_cast<int>(kron_matrices.size()));

  if (index == (static_cast<int>(kron_matrices.size()) - 1))
  {
    return fk::matrix<P>(kron_matrices.back());
  }

  else
  {
    return fk::matrix<P>(
        kron_matrices[index].kron(recursive_kron(kron_matrices, index + 1)));
  }
}

/* given a pde, for each dimension create a matrix where the columns are
   legendre basis functions evaluated at the roots */
template<typename P>
std::vector<fk::matrix<P>> gen_realspace_transform(
    PDE<P> const &pde,
    basis::wavelet_transform<P, resource::host> const &transformer)
{
  /* contains a basis matrix for each dimension */
  std::vector<fk::matrix<P>> real_space_transform;
  real_space_transform.reserve(pde.num_dims);

  for (int i = 0; i < pde.num_dims; i++)
  {
    /* get the ith dimension */
    dimension<P> const &d    = pde.get_dimensions()[i];
    int const level          = d.get_level();
    int const n_segments     = fm::two_raised_to(level);
    int const deg_freedom_1d = d.get_degree() * n_segments;
    P const normalize        = (d.domain_max - d.domain_min) / n_segments;
    fk::matrix<P> dimension_transform(deg_freedom_1d, deg_freedom_1d);
    /* create matrix of Legendre polynomial basis functions evaluated at the
     * roots */
    bool const use_degree_quad = true;
    auto const roots =
        legendre_weights<P>(d.get_degree(), -1, 1, use_degree_quad)[0];
    /* normalized legendre transformation matrix. Column i is legendre
       polynomial of degree i. element (i, j) is polynomial evaluated at jth
       root of the highest degree polynomial */
    fk::matrix<P> const basis = legendre<P>(roots, d.get_degree())[0] *
                                (static_cast<P>(1.0) / std::sqrt(normalize));
    /* set submatrices of dimension_transform */
    for (int j = 0; j < n_segments; j++)
    {
      int const diagonal_pos = d.get_degree() * j;
      dimension_transform.set_submatrix(diagonal_pos, diagonal_pos, basis);
    }
    real_space_transform.push_back(transformer.apply(dimension_transform, level,
                                                     basis::side::right,
                                                     basis::transpose::trans));
  }
  return real_space_transform;
}

template<typename P>
void wavelet_to_realspace(
    PDE<P> const &pde, fk::vector<P> const &wave_space,
    elements::table const &table,
    basis::wavelet_transform<P, resource::host> const &transformer,
    int const memory_limit_MB,
    std::array<fk::vector<P, mem_type::view, resource::host>, 2> &workspace,
    fk::vector<P> &real_space)
{
  expect(memory_limit_MB > 0);

  std::vector<batch_chain<P, resource::host>> chain;

  /* generate the wavelet-to-real-space transformation matrices for each
   * dimension */
  std::vector<fk::matrix<P>> real_space_transform =
      gen_realspace_transform(pde, transformer);

  // FIXME Assume the degree in the first dimension is equal across all the
  // remaining dimensions
  auto const degree = pde.get_dimensions()[0].get_degree();
  auto const stride = std::pow(degree, pde.num_dims);

  fk::vector<P, mem_type::owner, resource::host> accumulator(real_space.size());
  fk::vector<P, mem_type::view, resource::host> real_space_accumulator(
      accumulator);

  for (auto i = 0; i < table.size(); i++)
  {
    std::vector<fk::matrix<P, mem_type::const_view>> kron_matrices;
    kron_matrices.reserve(pde.num_dims);
    auto const coords = table.get_coords(i);

    for (auto j = 0; j < pde.num_dims; j++)
    {
      auto const id =
          elements::get_1d_index(coords(j), coords(j + pde.num_dims));
      auto const degree = pde.get_dimensions()[j].get_degree();
      fk::matrix<P, mem_type::const_view> sub_matrix(
          real_space_transform[j], 0, real_space_transform[j].nrows() - 1,
          id * degree, (id + 1) * degree - 1);
      kron_matrices.push_back(sub_matrix);
    }

    /* compute the amount of needed memory */
    expect(kron_matrix_MB(kron_matrices) <= memory_limit_MB);

    /* create a view of a section of the wave space vector */
    fk::vector<P, mem_type::const_view> const x(wave_space, i * stride,
                                                (i + 1) * stride - 1);

    chain.emplace_back(kron_matrices, x, workspace, real_space_accumulator);
  }

  /* clear out the vector */
  real_space.scale(0);

  for (auto const &link : chain)
  {
    link.execute();
    real_space = real_space + real_space_accumulator;
  }

  return;
}

// FIXME this function will need to change once dimensions can have different
// degree...
// combine components and create the portion of the multi-d vector associated
// with the provided start and stop element bounds (inclusive)
template<typename P>
fk::vector<P>
combine_dimensions(int const degree, elements::table const &table,
                   int const start_element, int const stop_element,
                   std::vector<fk::vector<P>> const &vectors,
                   P const time_scale)
{
  int const num_dims = vectors.size();
  expect(num_dims > 0);
  expect(start_element >= 0);
  expect(stop_element >= start_element);
  expect(stop_element < table.size());

  int64_t const vector_size =
      (stop_element - start_element + 1) * std::pow(degree, num_dims);

  // FIXME here we want to catch the 64-bit solution vector problem
  // and halt execution if we spill over. there is an open issue for this
  expect(vector_size < INT_MAX);
  fk::vector<P> combined(vector_size);

  for (int i = start_element; i <= stop_element; ++i)
  {
    std::vector<fk::vector<P>> kron_list;
    fk::vector<int> const coords = table.get_coords(i);
    for (int j = 0; j < num_dims; ++j)
    {
      // iterating over cell coords;
      // first num_dims entries in coords are level coords
      int const id = elements::get_1d_index(coords(j), coords(j + num_dims));
      int const index_start = id * degree;
      // index_start and index_end describe a subvector of length degree;
      // for deg = 1, this is a vector of one element
      int const index_end =
          degree > 1 ? (((id + 1) * degree) - 1) : index_start;
      kron_list.push_back(vectors[j].extract(index_start, index_end));
    }
    int const start_index = (i - start_element) * std::pow(degree, num_dims);
    int const stop_index  = start_index + std::pow(degree, num_dims) - 1;
    fk::vector<P, mem_type::view> combined_view(combined, start_index,
                                                stop_index);

    combined_view = kron_d(kron_list, kron_list.size()) * time_scale;
  }

  return combined;
}

/* explicit instantiations */
#define X(T)                                                     \
  template fk::matrix<T> recursive_kron(                         \
      std::vector<fk::matrix<T, mem_type::view>> &kron_matrices, \
      int const index);
#include "type_list_float.inc"
#undef X

#define X(T)                                                   \
  template std::vector<fk::matrix<T>> gen_realspace_transform( \
      PDE<T> const &pde,                                       \
      basis::wavelet_transform<T, resource::host> const &transformer);
#include "type_list_float.inc"
#undef X

#define X(T)                                                                   \
  template void wavelet_to_realspace(                                          \
      PDE<T> const &pde, fk::vector<T> const &wave_space,                      \
      elements::table const &table,                                            \
      basis::wavelet_transform<T, resource::host> const &transformer,          \
      int const memory_limit_MB,                                               \
      std::array<fk::vector<T, mem_type::view, resource::host>, 2> &workspace, \
      fk::vector<T> &real_space);
#include "type_list_float.inc"
#undef X

#define X(T)                                                    \
  template fk::vector<T> combine_dimensions(                    \
      int const, elements::table const &, int const, int const, \
      std::vector<fk::vector<T>> const &, T const = 1.0);
#include "type_list_float.inc"
#undef X
