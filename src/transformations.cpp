#include "transformations.hpp"
#include "connectivity.hpp"
#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include <algorithm>
#include <cassert>
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
  assert(num_prods > 0);
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
    std::vector<fk::matrix<P, mem_type::view>> const &kron_matrices)
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
  assert(index >= 0);
  assert(index < static_cast<int>(kron_matrices.size()));

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
std::vector<fk::matrix<P>> gen_realspace_transform(PDE<P> const &pde)
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
    auto const roots = legendre_weights<P>(d.get_degree(), -1, 1)[0];
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
    real_space_transform.push_back(dimension_transform *
                                   fk::matrix<P>(d.get_from_basis_operator()));
  }
  return real_space_transform;
}

template<typename P>
fk::vector<P>
wavelet_to_realspace(PDE<P> const &pde, fk::vector<P> const &wave_space,
                     element_table const &table, int const memory_limit_MB)
{
  assert(memory_limit_MB > 0);

  std::vector<dimension<P>> const &dims = pde.get_dimensions();
  /* determine the length of the real-space vector */
  int prod = 1;
  for (int i = 0; i < static_cast<int>(dims.size()); i++)
  {
    prod *= (dims[i].get_degree() * std::pow(2, dims[i].get_level()));
  }

  fk::vector<P> real_space(prod);
  /* generate the wavelet-to-real-space transformation matrices for each
   * dimension */
  std::vector<fk::matrix<P>> real_space_transform =
      gen_realspace_transform(pde);
  /* Assume the degree in the first dimension is equal across all the remaining
   */
  int const stride =
      std::pow(pde.get_dimensions()[0].get_degree(), dims.size());

  for (int i = 0; i < table.size(); i++)
  {
    std::vector<fk::matrix<P, mem_type::view>> kron_matrices;
    kron_matrices.reserve(pde.num_dims);
    fk::vector<int> const coords = table.get_coords(i);

    for (int j = 0; j < pde.num_dims; j++)
    {
      int const id     = get_1d_index(coords(j), coords(j + pde.num_dims));
      int const degree = pde.get_dimensions()[j].get_degree();
      fk::matrix<P, mem_type::view> sub_matrix(
          real_space_transform[j], 0, real_space_transform[j].nrows() - 1,
          id * degree, (id + 1) * degree - 1);
      kron_matrices.push_back(sub_matrix);
    }

    /* compute the amount of needed memory */
    assert(kron_matrix_MB(kron_matrices) <= memory_limit_MB);

    /* get a matrix by kronecker producting the list together */
    fk::matrix<P> const kronecker_product = recursive_kron(kron_matrices);
    fk::vector<P, mem_type::view> const x(wave_space, i * stride,
                                          (i + 1) * stride - 1);
    /* add it to the realspace vector */
    real_space = real_space + (kronecker_product * x);
  }

  return real_space;
}

// FIXME this function will need to change once dimensions can have different
// degree...
// combine components and create the portion of the multi-d vector associated
// with the provided start and stop element bounds (inclusive)
template<typename P>
fk::vector<P>
combine_dimensions(int const degree, element_table const &table,
                   int const start_element, int const stop_element,
                   std::vector<fk::vector<P>> const &vectors,
                   P const time_scale)
{
  int const num_dims = vectors.size();
  assert(num_dims > 0);
  assert(start_element >= 0);
  assert(stop_element >= start_element);
  assert(stop_element < table.size());

  int64_t const vector_size =
      (stop_element - start_element + 1) * std::pow(degree, num_dims);

  // FIXME here we want to catch the 64-bit solution vector problem
  // and halt execution if we spill over. there is an open issue for this
  assert(vector_size < INT_MAX);
  fk::vector<P> combined(vector_size);

  for (int i = start_element; i <= stop_element; ++i)
  {
    std::vector<fk::vector<P>> kron_list;
    fk::vector<int> const coords = table.get_coords(i);
    for (int j = 0; j < num_dims; ++j)
    {
      // iterating over cell coords;
      // first num_dims entries in coords are level coords
      int const id          = get_1d_index(coords(j), coords(j + num_dims));
      int const index_start = id * degree;
      // index_start and index_end describe a subvector of length degree;
      // for deg = 1, this is a vector of one element
      int const index_end =
          degree > 1 ? (((id + 1) * degree) - 1) : index_start;
      kron_list.push_back(vectors[j].extract(index_start, index_end));
    }
    fk::vector<P> const partial_result =
        kron_d(kron_list, kron_list.size()) * time_scale;
    combined.set_subvector((i - start_element) * std::pow(degree, num_dims),
                           partial_result);
  }
  return combined;
}

/* explicit instantiations */
template fk::matrix<double>
recursive_kron(std::vector<fk::matrix<double, mem_type::view>> &kron_matrices,
               int const index);

template fk::matrix<float>
recursive_kron(std::vector<fk::matrix<float, mem_type::view>> &kron_matrices,
               int const index);

template std::vector<fk::matrix<double>>
gen_realspace_transform(PDE<double> const &pde);

template std::vector<fk::matrix<float>>
gen_realspace_transform(PDE<float> const &pde);

template fk::vector<double>
wavelet_to_realspace(PDE<double> const &pde,
                     fk::vector<double> const &wave_space,
                     element_table const &table, int const memory_limit_MB);
template fk::vector<float>
wavelet_to_realspace(PDE<float> const &pde, fk::vector<float> const &wave_space,
                     element_table const &table, int const memory_limit_MB);

template fk::vector<double>
combine_dimensions(int const, element_table const &, int const, int const,
                   std::vector<fk::vector<double>> const &, double const = 1.0);
template fk::vector<float>
combine_dimensions(int const, element_table const &, int const, int const,
                   std::vector<fk::vector<float>> const &, float const = 1.0);
