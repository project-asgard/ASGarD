#include "basis.hpp"
#include "distribution.hpp"
#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "quadrature.hpp"
#include "tensors.hpp"
#include "tools.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

// generate_multi_wavelets routine creates wavelet basis (phi_co)
// then uses these to generate the two-scale coefficients which can be
// used (outside of this routine) to construct the forward multi-wavelet
// transform
template<typename P>
std::array<fk::matrix<P>, 6> generate_multi_wavelets(int const degree)
{
  expect(degree > 0);

  // These are the function outputs
  // g0,g1,h0, and h1 are two-scale coefficients
  // The returned phi_co is the wavelet basis
  // scalet_coefficients are the scaling function basis
  fk::matrix<P> g0(degree, degree);
  fk::matrix<P> g1(degree, degree);
  fk::matrix<P> h0(degree, degree);
  fk::matrix<P> h1(degree, degree);
  fk::matrix<P> phi_co(degree * 2, degree);
  fk::matrix<P> scalet_coefficients(degree, degree);

  // Set up parameters for quadrature - used for inner product evaluation
  // In this routine there are three intervals for the quadrature
  // Negative one to zero --zero to one -- and negative one to one
  // since this is done on the "reference element" -1 to 1
  // the intervals are fixed to be (-1,0), (0,1), and (-1,1)
  P const neg1 = -1.0;
  P const zero = 0.0;
  P const one  = 1.0;

  // Step 1 of 2 in creating scalets
  // generate Legendre polynomial coefficients
  // size_legendre degree+2 is used so that this algorithm will
  // work with degree 0
  // the recurrence relation used here can be found at
  // http://mathworld.wolfram.com/LegendrePolynomial.html equation (43)
  int const size_legendre = degree + 2;
  fk::matrix<P> legendre(size_legendre, size_legendre);
  legendre(0, size_legendre - 1) = 1.0;
  legendre(1, size_legendre - 2) = 1.0;
  for (int row_pos = 0; row_pos < degree - 2; ++row_pos)
  {
    // The Legendre polynomial of order row_pos+2 is constructed
    // from Legendre polynomial order row_pos and row_pos+1
    // single row matrix; using matrix here for set submatrix
    fk::matrix<P> const P_row_pos =
        legendre.extract_submatrix(row_pos + 1, 0, 1, size_legendre);
    fk::matrix<P> const P_row_pos_minus1 =
        legendre.extract_submatrix(row_pos, 0, 1, size_legendre);

    // Shifting in matrix form is equivalent to multiplying by x
    fk::matrix<P> row_shift(1, size_legendre);
    row_shift.set_submatrix(
        0, size_legendre - row_pos - 3,
        P_row_pos.extract_submatrix(0, size_legendre - row_pos - 2, 1,
                                    row_pos + 2));
    legendre.update_row(
        row_pos + 2,
        fk::vector<P>(
            row_shift *
                ((2.0 * (row_pos + 1.0) + 1.0) / ((row_pos + 1.0) + 1.0)) -
            P_row_pos_minus1 * ((row_pos + 1.0) / ((row_pos + 1.0) + 1.0))));
  }

  // Step 2 of 2 in creating the scalets
  // scale the matrix phi_j = sqrt(2*j+1)*P_j(2*x-1)
  // where P_j is Legendre polynomial of degree j
  for (int row_pos = 0; row_pos < size_legendre; ++row_pos)
  {
    fk::vector<P> const row =
        legendre.extract_submatrix(row_pos, 0, 1, size_legendre);
    legendre.update_row(row_pos, row * std::sqrt(2.0 * (row_pos) + 1));
  }

  // Roots and weights of Legendre polynomials on (-1,0), (0,1), and (-1,1)
  // we do the two-step store because we cannot have 'static' bindings
  auto const [roots_neg1to0, weights_neg1to0] =
      legendre_weights<P>(degree, neg1, zero);

  auto const [roots_0to1, weights_0to1] =
      legendre_weights<P>(degree, zero, one);

  auto const [roots_neg1to1, weights_neg1to1] =
      legendre_weights<P>(degree, neg1, one);

  // this is to get around unused warnings
  // because can't unpack only some args w structured binding (until c++20)
  auto const ignore = [](auto ignored) { (void)ignored; };
  ignore(weights_neg1to1);

  auto const get_row = [](fk::matrix<P> const mat,
                          int const row_pos) -> fk::vector<P> {
    return mat.extract_submatrix(row_pos, 0, 1, mat.ncols());
  };

  // Formulation of phi_co before orthogonalization begins
  // phi_co is the coefficients of the polynomials
  //        f_j = |  x^(j-1) on (0,1)
  //              | -x^(j-1) on (-1,0)
  fk::matrix<P> const norm_co = [&] {
    fk::matrix<P> flip_identity = eye<P>(degree);
    for (int i = 0; i < degree; ++i)
    {
      fk::vector<int> row = fk::vector<int>(get_row(flip_identity, i));
      std::reverse(row.begin(), row.end());
      flip_identity.update_row(i, fk::vector<P>(row));
    }
    return flip_identity;
  }();

  scalet_coefficients.set_submatrix(
      0, 0,
      legendre.extract_submatrix(0, size_legendre - degree, degree, degree));

  phi_co.set_submatrix(0, 0, norm_co * -1);
  phi_co.set_submatrix(degree, 0, norm_co);

  auto const weighted_sum_products = [&](fk::vector<P> const vect_1,
                                         fk::vector<P> const vect_2,
                                         fk::vector<P> const weights) -> P {
    P sum = 0.0;
    for (int i = 0; i < weights.size(); ++i)
    {
      sum += vect_1(i) * vect_2(i) * weights(i);
    }
    return sum * 0.5;
  };

  // Gram-Schmidt orthogonalization of phi_co
  // with respect to scalets
  // The inner products are taken of:
  // elem_1 - phi_co from -1 to 0
  // elem_2 - scalet_coefficients (Legendre Poly.) from -1 to 0
  // elem_3 - phi_co from 0 to 1
  // elem_4 - scalet_coefficients (Legendre Poly.) from 0 to 1
  for (int row_pos = 0; row_pos < degree; ++row_pos)
  {
    fk::vector<P> proj(degree);
    for (int row_2 = 0; row_2 < degree; ++row_2)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(phi_co, row_pos), roots_neg1to0);
      fk::vector<P> const elem_2 =
          polyval(get_row(scalet_coefficients, row_2), roots_neg1to0);
      fk::vector<P> const elem_3 =
          polyval(get_row(phi_co, row_pos + degree), roots_0to1);
      fk::vector<P> const elem_4 =
          polyval(get_row(scalet_coefficients, row_2), roots_0to1);

      P const sum = weighted_sum_products(elem_1, elem_2, weights_neg1to0) +
                    weighted_sum_products(elem_3, elem_4, weights_0to1);
      proj = proj + (get_row(scalet_coefficients, row_2) * sum);
    }
    phi_co.update_row(row_pos, (get_row(phi_co, row_pos) - proj));
    phi_co.update_row(row_pos + degree,
                      (get_row(phi_co, row_pos + degree) - proj));
  }

  // Gram-Schmidt orthogonalization of phi_co
  // with respect to each degree less than itself
  // e.g. row k is orthogonalized with respect to rows k-1, k-2 to zero
  // row k+degree is orthogonalized with respect to rows
  // k+degree-1, k+degree-2, to degree
  // The inner products are taken of:
  // elem_1 - phi_co from -1 to 0 of order k
  // elem_2 - phi_co from -1 to 0 of order k-1
  // elem_3 - phi_co from 0 to 1 - of order k
  // elem_4 - phi_co from 0 to 1 - of order k-1
  // sum_1 is the inner product of polynomial degree k
  //    with polynomial degree k-1
  // sum_2 is the norm of polynomial degree k-1
  for (int row_pos = 1; row_pos < degree; ++row_pos)
  {
    fk::vector<P> proj_1(degree);
    fk::vector<P> proj_2(degree);
    for (int row_2 = 0; row_2 < row_pos; ++row_2)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(phi_co, row_pos), roots_neg1to0);
      fk::vector<P> const elem_2 =
          polyval(get_row(phi_co, row_2), roots_neg1to0);
      fk::vector<P> const elem_3 =
          polyval(get_row(phi_co, row_pos + degree), roots_0to1);
      fk::vector<P> const elem_4 =
          polyval(get_row(phi_co, row_2 + degree), roots_0to1);

      P const sum_1 = weighted_sum_products(elem_1, elem_2, weights_neg1to0) +
                      weighted_sum_products(elem_3, elem_4, weights_0to1);

      P const sum_2 = weighted_sum_products(elem_2, elem_2, weights_neg1to0) +
                      weighted_sum_products(elem_4, elem_4, weights_0to1);

      fk::vector<P> proj_1_add = get_row(phi_co, row_2) * sum_1;
      fk::vector<P> proj_2_add = get_row(phi_co, row_2 + degree) * sum_1;
      for (int i = 0; i < proj_1_add.size(); ++i)
      {
        proj_1_add(i) /= sum_2;
        proj_2_add(i) /= sum_2;
      }
      proj_1 = proj_1 + proj_1_add;
      proj_2 = proj_2 + proj_2_add;
    }

    phi_co.update_row(row_pos, (get_row(phi_co, row_pos) - proj_1));
    phi_co.update_row(row_pos + degree,
                      (get_row(phi_co, row_pos + degree) - proj_2));
  }

  // Normalization of phi_co basis vectors
  for (int row_pos = 0; row_pos < degree; ++row_pos)
  {
    fk::vector<P> const elem_1 =
        polyval(get_row(phi_co, row_pos), roots_neg1to0);
    fk::vector<P> const elem_2 =
        polyval(get_row(phi_co, row_pos + degree), roots_0to1);

    P const sum = weighted_sum_products(elem_1, elem_1, weights_neg1to0) +
                  weighted_sum_products(elem_2, elem_2, weights_0to1);

    fk::vector<P> row   = get_row(phi_co, row_pos);
    fk::vector<P> row_2 = get_row(phi_co, row_pos + degree);

    for (int i = 0; i < row.size(); ++i)
    {
      row(i) /= std::sqrt(sum);
      row_2(i) /= std::sqrt(sum);
    }

    phi_co.update_row(row_pos, row);
    phi_co.update_row(row_pos + degree, row_2);
  }

  // build a degree by degree matrix with alternating rows
  // of -1.0 and 1.0, then stack it vertically
  // This restores sign of polynomials after normalization
  fk::matrix<P> rep_mat(2 * degree, degree);
  std::vector<P> const pos_one(degree, 1.0);
  std::vector<P> const neg_one(degree, -1.0);
  for (int i = 0; i < degree; ++i)
  {
    if ((i % 2) == 0)
    {
      rep_mat.update_row(i, pos_one);
      rep_mat.update_row(i + degree, pos_one);
    }
    else
    {
      rep_mat.update_row(i, neg_one);
      rep_mat.update_row(i + degree, neg_one);
    }
  }

  // reverse the rows of degree by degree
  // vertical slices of phi_co
  for (int i = 0; i < 2; ++i)
  {
    fk::matrix<P> phi_co_part =
        phi_co.extract_submatrix(i * degree, 0, degree, degree);
    fk::matrix<P> const phi_co_part_copy(phi_co_part);
    for (int j = 0; j < phi_co_part.nrows(); ++j)
    {
      fk::vector<P> const new_row = phi_co_part_copy.extract_submatrix(
          phi_co_part.nrows() - 1 - j, 0, 1, phi_co_part.ncols());
      phi_co_part.update_row(j, new_row);
    }
    phi_co.set_submatrix(i * degree, 0, phi_co_part);
  }

  // finally, multiply the two elementwise
  std::transform(phi_co.begin(), phi_co.end(), rep_mat.begin(), phi_co.begin(),
                 [](P &elem_1, P &elem_2) { return elem_1 * elem_2; });

  // Calculate Two-Scale Coefficients

  // Sums to directly generate H0, H1, G0, G1
  //  H0 and H1 are the "coarsening coefficients"
  //  These describe how two adjacent locations of a higher (finer resolution)
  //  level sum to give a lower (more coarse resolution) level coefficients
  //  G0 and G1 are the "refining or detail coefficients"
  //  These describe how lower level (more coarse resolution)
  //  is split into two higher (finer resolution) level coefficients
  //  H0 is the inner product of the scaling functions of two successive
  //   levels - thus the difference in roots
  // elem_1 is the scalet functions on (-1,0)
  // elem_2 is the scalet function of a lower level and therefore spans (-1,1)
  //  H1 is also the inner product of the scaling functions of two successive
  //   levels - thus the difference in roots
  // elem_3 is the scalet functions on (0,1)
  //  G0 is the inner product of the wavelet functions of one level
  //   with the scalet functions of a lower level
  //   - thus the difference in roots
  // elem_4 is the wavelet functions on (-1,0)
  //  G1 is also the inner product of the wavelet functions of one level
  //   with the scalet functions of a lower level
  // elem_5 is the scalet functions on (0,1)

  for (int row = 0; row < degree; ++row)
  {
    for (int col = 0; col < degree; ++col)
    {
      fk::vector<P> const elem_1 =
          polyval(get_row(scalet_coefficients, row), roots_neg1to0);
      fk::vector<P> const elem_2 =
          polyval(get_row(scalet_coefficients, col), roots_neg1to1);
      h0(row, col) = 2.0 *
                     weighted_sum_products(elem_1, elem_2, weights_neg1to0) /
                     std::sqrt(2.0);

      fk::vector<P> const elem_3 =
          polyval(get_row(scalet_coefficients, row), roots_0to1);
      h1(row, col) = 2.0 * weighted_sum_products(elem_3, elem_2, weights_0to1) /
                     std::sqrt(2.0);

      fk::vector<P> const elem_4 = polyval(get_row(phi_co, row), roots_neg1to0);
      g0(row, col)               = 2.0 *
                     weighted_sum_products(elem_4, elem_2, weights_neg1to0) /
                     std::sqrt(2.0);

      fk::vector<P> const elem_5 =
          polyval(get_row(phi_co, row + degree), roots_0to1);
      g1(row, col) = 2.0 * weighted_sum_products(elem_5, elem_2, weights_0to1) /
                     std::sqrt(2.0);
    }
  }

  P const compare = [] {
    if constexpr (std::is_same<P, double>::value)
    {
      return static_cast<P>(1e-12);
    }
    return static_cast<P>(1e-4);
  }();
  auto const normalize = [compare](fk::matrix<P> &matrix) {
    std::transform(
        matrix.begin(), matrix.end(), matrix.begin(),
        [compare](P &elem) { return std::abs(elem) < compare ? 0.0 : elem; });
  };
  normalize(h0);
  normalize(h1);
  normalize(g0);
  normalize(g1);
  normalize(phi_co);
  normalize(scalet_coefficients);

  return std::array<fk::matrix<P>, 6>{h0, h1,     g0,
                                      g1, phi_co, scalet_coefficients};
}

template<typename R>
fk::matrix<R> operator_two_scale(int const degree, int const num_levels)
{
  expect(degree > 0);
  expect(num_levels > 1);

  int const max_level = fm::two_raised_to(num_levels);

  // this is to get around unused warnings
  // because can't unpack only some args w structured binding (until c++20)
  auto const ignore = [](auto ignored) { (void)ignored; };
  auto const [h0, h1, g0, g1, phi_co, scale_co] =
      generate_multi_wavelets<R>(degree);
  ignore(phi_co);
  ignore(scale_co);

  fk::matrix<R> fmwt(degree * max_level, degree * max_level);

  fk::matrix<R> const h_block = fk::matrix<R>(h0.nrows(), h0.ncols() * 2)
                                    .set_submatrix(0, 0, h0)
                                    .set_submatrix(0, h0.ncols(), h1);
  fk::matrix<R> const g_block = fk::matrix<R>(g0.nrows(), g0.ncols() * 2)
                                    .set_submatrix(0, 0, g0)
                                    .set_submatrix(0, g0.ncols(), g1);

  // set the top vertical half of fmwt along the block diagonal with h_block
  // the lower half is set in the same manner, but with g_block
  for (int i = 0; i < max_level / 2; ++i)
  {
    fmwt.set_submatrix(degree * i, 2 * degree * i, h_block);
    fmwt.set_submatrix(degree * (i + max_level / 2), 2 * degree * i, g_block);
  }

  fk::matrix<R> fmwt_comp = eye<R>(degree * max_level, degree * max_level);

  int const n = std::floor(std::log2(max_level));
  for (int j = 1; j <= n; j++)
  {
    fk::matrix<R> cfmwt(degree * max_level, degree * max_level);
    if (j == 1)
    {
      cfmwt = fmwt;
    }
    else
    {
      int const cn = fm::two_raised_to(n - j + 1) * degree;

      std::fill(cfmwt.begin(), cfmwt.end(), 0.0);
      cfmwt.set_submatrix(cn, cn, eye<R>(degree * max_level - cn));
      cfmwt.set_submatrix(
          0, 0,
          fk::matrix<R, mem_type::const_view>(fmwt, 0, cn / 2 - 1, 0, cn - 1));
      cfmwt.set_submatrix(cn / 2, 0,
                          fk::matrix<R, mem_type::const_view>(
                              fmwt, degree * max_level / 2,
                              degree * max_level / 2 + cn / 2 - 1, 0, cn - 1));
    }
    fmwt_comp = cfmwt * fmwt_comp;
  }
  std::transform(fmwt_comp.begin(), fmwt_comp.end(), fmwt_comp.begin(),
                 [](R &elem) { return std::abs(elem) < 1e-12 ? 0.0 : elem; });
  return fmwt_comp;
}

template std::array<fk::matrix<double>, 6>
generate_multi_wavelets(int const degree);
template std::array<fk::matrix<float>, 6>
generate_multi_wavelets(int const degree);

template fk::matrix<double>
operator_two_scale(int const degree, int const num_levels);
template fk::matrix<float>
operator_two_scale(int const degree, int const num_levels);

namespace basis
{
// FIXME assumes same degree for all dimensions
template<typename P, resource resrc>
wavelet_transform<P, resrc>::wavelet_transform(options const &program_opts,
                                               PDE<P> const &pde,
                                               bool const quiet)
    : max_level(program_opts.max_level),
      degree(pde.get_dimensions()[0].get_degree()), dense_blocks_(max_level * 2)
{
  // this is to get around unused warnings
  // because can't unpack only some args w structured binding (until
  // c++20)
  auto const ignore = [](auto ignored) { (void)ignored; };
  auto const [h0, h1, g0, g1, phi_co, scale_co] =
      generate_multi_wavelets<P>(degree);
  ignore(phi_co);
  ignore(scale_co);

  auto const fmwt_size = degree * fm::two_raised_to(max_level);

  std::vector<fk::matrix<P>> block_builder(max_level * 2);

  fk::matrix<P> g_mat(degree, fmwt_size);
  fk::matrix<P> h_mat = fk::matrix<P>(degree, fmwt_size)
                            .set_submatrix(0, 0, eye<P>(degree, degree));

  // main loop - build the blocks with small gemms
  for (auto j = max_level - 1; j >= 0; --j)
  {
    auto const num_cells   = fm::two_raised_to(j);
    auto const block_ncols = fmwt_size / num_cells;
    auto const ncols_h     = block_ncols / 2;

    fk::matrix<P> h_tmp(degree, ncols_h);
    fk::matrix<P, mem_type::view> h_view(h_mat, 0, degree - 1, 0, ncols_h - 1);
    h_tmp = h_view;

    fk::matrix<P, mem_type::view> g_view(g_mat, 0, degree - 1, 0, ncols_h - 1);
    fm::gemm(g0, h_tmp, g_view);

    fk::matrix<P, mem_type::view> g_view_2(g_mat, 0, degree - 1, ncols_h,
                                           ncols_h * 2 - 1);
    fm::gemm(g1, h_tmp, g_view_2);

    fm::gemm(h0, h_tmp, h_view);

    fk::matrix<P, mem_type::view> h_view_2(h_mat, 0, degree - 1, ncols_h,
                                           ncols_h * 2 - 1);
    fm::gemm(h1, h_tmp, h_view_2);

    fk::matrix<P, mem_type::const_view> const g_block(g_mat, 0, degree - 1, 0,
                                                      block_ncols - 1);
    block_builder[j * 2 + 1].clear_and_resize(degree, block_ncols) = g_block;

    fk::matrix<P, mem_type::const_view> const h_block(h_mat, 0, degree - 1, 0,
                                                      block_ncols - 1);
    block_builder[j * 2].clear_and_resize(degree, block_ncols) = h_block;
  }

  // how much space are we using?
  auto const num_elems =
      std::accumulate(block_builder.begin(), block_builder.end(), 0,
                      [](int64_t const sum, auto const &matrix) {
                        return sum + static_cast<int64_t>(matrix.size());
                      });

  if (!quiet)
    node_out() << "  basis operator allocation (MB): " << get_MB<P>(num_elems)
               << '\n';

  // copy to device if necessary
  expect(block_builder.size() == dense_blocks_.size());
  for (auto i = 0; i < static_cast<int>(block_builder.size()); ++i)
  {
    if constexpr (resrc == resource::host)
    {
      dense_blocks_[i].clear_and_resize(block_builder[i].nrows(),
                                        block_builder[i].ncols()) =
          block_builder[i];
    }
    else
    {
      dense_blocks_[i]
          .clear_and_resize(block_builder[i].nrows(), block_builder[i].ncols())
          .transfer_from(block_builder[i]);
    }
  }
}

template<typename P, resource resrc>
template<mem_type omem>
fk::vector<P, mem_type::owner, resrc> wavelet_transform<P, resrc>::apply(
    fk::vector<P, omem, resrc> const &coefficients, int const level,
    basis::side const transform_side,
    basis::transpose const transform_trans) const
{
  expect(level >= 0);
  expect(level <= max_level);

  auto const ncols =
      transform_side == basis::side::right ? coefficients.size() : 1;
  auto const nrows =
      transform_side == basis::side::right ? 1 : coefficients.size();
  auto const as_matrix = apply(
      fk::matrix<P, mem_type::const_view, resrc>(coefficients, nrows, ncols, 0),
      level, transform_side, transform_trans);
  return fk::vector<P, mem_type::owner, resrc>(as_matrix);
}
template<typename P, resource resrc>
template<mem_type omem>
fk::matrix<P, mem_type::owner, resrc> wavelet_transform<P, resrc>::apply(
    fk::matrix<P, omem, resrc> const &coefficients, int const level,
    basis::side const transform_side,
    basis::transpose const transform_trans) const
{
  expect(level >= 0);
  if (level == 0)
  {
    return fk::matrix<P, mem_type::owner, resrc>(coefficients);
  }

  expect(level <= max_level);

  auto const op_size = fm::two_raised_to(level) * degree;
  if (transform_side == basis::side::right)
  {
    expect(coefficients.ncols() == op_size);
  }
  else
  {
    expect(coefficients.nrows() == op_size);
  }

  int const rows_y =
      transform_side == basis::side::left ? op_size : coefficients.nrows();
  int const cols_y =
      transform_side == basis::side::left ? coefficients.ncols() : op_size;
  fk::matrix<P, mem_type::owner, resrc> transformed(rows_y, cols_y);

  // first, coarsest level
  auto const do_trans =
      transform_trans == basis::transpose::trans ? true : false;
  auto const &first_block = dense_blocks_[(max_level - level) * 2];

  if (transform_side == basis::side::left)
  {
    if (transform_trans == basis::transpose::trans)
    {
      fk::matrix<P, mem_type::const_view, resrc> const B(
          coefficients, 0, degree - 1, 0, coefficients.ncols() - 1);
      fk::matrix<P, mem_type::view, resrc> C(transformed, 0, op_size - 1, 0,
                                             coefficients.ncols() - 1);

      fm::gemm(first_block, B, C, do_trans);
    }
    else
    {
      fk::matrix<P, mem_type::const_view, resrc> const B(
          coefficients, 0, op_size - 1, 0, coefficients.ncols() - 1);
      fk::matrix<P, mem_type::view, resrc> C(transformed, 0, degree - 1, 0,
                                             coefficients.ncols() - 1);

      fm::gemm(first_block, B, C, do_trans);
    }
  }
  else
  {
    if (transform_trans == basis::transpose::trans)
    {
      fk::matrix<P, mem_type::const_view, resrc> const A(
          coefficients, 0, coefficients.nrows() - 1, 0, op_size - 1);
      fk::matrix<P, mem_type::view, resrc> C(
          transformed, 0, coefficients.nrows() - 1, 0, degree - 1);

      fm::gemm(A, first_block, C, false, do_trans);
    }
    else
    {
      fk::matrix<P, mem_type::const_view, resrc> const A(
          coefficients, 0, coefficients.nrows() - 1, 0, degree - 1);
      fk::matrix<P, mem_type::view, resrc> C(
          transformed, 0, coefficients.nrows() - 1, 0, op_size - 1);

      fm::gemm(A, first_block, C, false, do_trans);
    }
  }

  // remaining levels
  auto const block_offset = (max_level - level) * 2 + 1;
  auto degree_start       = degree;

  for (auto i = 0; i < level; ++i)
  {
    auto const num_cells = fm::two_raised_to(i);
    auto const cell_size = op_size / num_cells;

    auto const &current_block = dense_blocks_[block_offset + i * 2];

    P const alpha = 1.0;
    P const beta  = 1.0;

    for (auto j = 0; j < num_cells; ++j)
    {
      auto const cell_start = j * cell_size;
      auto const cell_end   = cell_start + cell_size - 1;
      auto const degree_end = degree_start + degree - 1;

      if (transform_side == basis::side::left)
      {
        if (transform_trans == basis::transpose::trans)
        {
          fk::matrix<P, mem_type::view, resrc> C(
              transformed, cell_start, cell_end, 0, coefficients.ncols() - 1);
          fk::matrix<P, mem_type::const_view, resrc> const B(
              coefficients, degree_start, degree_end, 0,
              coefficients.ncols() - 1);

          fm::gemm(current_block, B, C, do_trans, false, alpha, beta);
        }
        else
        {
          fk::matrix<P, mem_type::view, resrc> C(transformed, degree_start,
                                                 degree_end, 0,
                                                 coefficients.ncols() - 1);
          fk::matrix<P, mem_type::const_view, resrc> const B(
              coefficients, cell_start, cell_end, 0, coefficients.ncols() - 1);

          fm::gemm(current_block, B, C, do_trans, false, alpha, beta);
        }
      }
      else
      {
        if (transform_trans == basis::transpose::trans)
        {
          fk::matrix<P, mem_type::view, resrc> C(transformed, 0,
                                                 coefficients.nrows() - 1,
                                                 degree_start, degree_end);
          fk::matrix<P, mem_type::const_view, resrc> const A(
              coefficients, 0, coefficients.nrows() - 1, cell_start, cell_end);
          fm::gemm(A, current_block, C, false, do_trans, alpha, beta);
        }
        else
        {
          fk::matrix<P, mem_type::view, resrc> C(
              transformed, 0, coefficients.nrows() - 1, cell_start, cell_end);
          fk::matrix<P, mem_type::const_view, resrc> const A(
              coefficients, 0, coefficients.nrows() - 1, degree_start,
              degree_end);

          fm::gemm(A, current_block, C, false, do_trans, alpha, beta);
        }
      }
      degree_start = degree_end + 1;
    }
  }

  return transformed;
}

#define X(T, Y) template class wavelet_transform<T, resource::Y>;
#include "type_list_float_view.inc"
#undef X

// float/double host/device owner/const_view -- No device + const_view
#define X(T, Y, Z)                                                 \
  template fk::vector<T, mem_type::owner, resource::Y>             \
  wavelet_transform<T, resource::Y>::apply(                        \
      fk::vector<T, mem_type::Z, resource::Y> const &coefficients, \
      int const level, basis::side const transform_side,           \
      basis::transpose const transform_trans) const;
#include "type_list_view.inc"
#undef X

// float/double host/device owner/const_view -- No device + const_view
#define X(T, Y, Z)                                                 \
  template fk::matrix<T, mem_type::owner, resource::Y>             \
  wavelet_transform<T, resource::Y>::apply(                        \
      fk::matrix<T, mem_type::Z, resource::Y> const &coefficients, \
      int const level, basis::side const transform_side,           \
      basis::transpose const transform_trans) const;
#include "type_list_view.inc"
#undef X
// fk::vector<float, (mem_type)0, (resource)0> basis::wavelet_transform<float,
// (resource)0>::apply<(mem_type)0>(fk::vector<float, (mem_type)0, (resource)0>
// const&, int, basis::side, basis::transpose) const'

} // namespace basis
