#include "quadrature.hpp"
#include "matlab_utilities.hpp"
#include "tools.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
// Evaluate Legendre polynomials on an input domain, trimmed to [-1,1]
// Virtually a direct translation of Ed's dlegendre2.m code
//
// Legendre matrix returned in the std::array[0], its derivative returned in [1]
// FIXME (is this right?) Each column of the matrix is such a polynomial (or its
// derivative) for each degree

template<typename P>
std::enable_if_t<std::is_floating_point<P>::value, std::array<fk::matrix<P>, 2>>
legendre(fk::vector<P> const domain, int const degree,
         legendre_normalization const normalization)
{
  expect(degree >= 0);
  expect(domain.size() > 0);

  // allocate and zero the output Legendre polynomials, their derivatives
  fk::matrix<P> legendre(domain.size(), std::max(1, degree));
  fk::matrix<P> legendre_prime(domain.size(), std::max(1, degree));

  legendre.update_col(0, std::vector<P>(domain.size(), static_cast<P>(1.0)));

  if (degree >= 2)
  {
    legendre.update_col(1, domain);
    legendre_prime.update_col(
        1, std::vector<P>(domain.size(), static_cast<P>(1.0)));
  }

  // if we are working to update column "n", then "_order" is the previous
  // column (i.e. n-1), and "_(order + 1)" is the one before that
  if (degree >= 3)
  {
    // initial values for n-1, n-2
    fk::vector<P> legendre_order =
        legendre.extract_submatrix(0, 1, domain.size(), 1);
    fk::vector<P> legendre_prime_order =
        legendre_prime.extract_submatrix(0, 1, domain.size(), 1);
    fk::vector<P> legendre_n_1 =
        legendre.extract_submatrix(0, 0, domain.size(), 1);
    fk::vector<P> legendre_prime_n_1 =
        legendre_prime.extract_submatrix(0, 0, domain.size(), 1);

    // set remaining columns
    for (int i = 0; i < (degree - 2); ++i)
    {
      int const n            = i + 1;
      int const column_index = i + 2;

      // element-wise multiplication
      fk::vector<P> product(domain.size());
      std::transform(domain.begin(), domain.end(), legendre_order.begin(),
                     product.begin(), std::multiplies<P>());

      P const factor = 1.0 / (n + 1.0);

      fk::vector<P> legendre_col = (product * static_cast<P>(2.0 * n + 1.0)) -
                                   (legendre_n_1 * static_cast<P>(n));
      legendre_col = legendre_col * factor;
      legendre.update_col(column_index, legendre_col);

      std::transform(domain.begin(), domain.end(), legendre_prime_order.begin(),
                     product.begin(), std::multiplies<P>());

      fk::vector<P> legendre_prime_col =
          (product + legendre_order) * static_cast<P>(2.0 * n + 1.0) -
          legendre_prime_n_1 * static_cast<P>(n);
      legendre_prime_col = legendre_prime_col * factor;
      legendre_prime.update_col(column_index, legendre_prime_col);

      // update columns for next iteration
      legendre_n_1         = legendre_order;
      legendre_order       = legendre_col;
      legendre_prime_n_1   = legendre_prime_order;
      legendre_prime_order = legendre_prime_col;
    }
  }

  // "normalizing"
  for (int i = 0; i < degree; ++i)
  {
    P dscale = -1;
    if (normalization == legendre_normalization::unnormalized)
    {
      dscale = static_cast<P>(1.0);
    }
    else if (normalization == legendre_normalization::lin)
    {
      P norm_2 = static_cast<P>(2.0) / (2.0 * i + 1.0);
      dscale   = static_cast<P>(1.0) / std::sqrt(norm_2);
    }
    else if (normalization == legendre_normalization::matlab)
    {
      dscale = static_cast<P>(1.0) / std::sqrt(2.0);
    }

    expect(dscale > 0);

    fk::vector<P> const legendre_sub =
        legendre.extract_submatrix(0, i, domain.size(), 1);
    legendre.update_col(i, legendre_sub * dscale);

    fk::vector<P> const legendre_prime_sub =
        legendre_prime.extract_submatrix(0, i, domain.size(), 1);
    legendre_prime.update_col(i, legendre_prime_sub * dscale);
  }

  // "zero out points out of range"
  fk::vector<int> const out_of_range = find(domain, [](P const &elem) {
    return elem < static_cast<P>(-1.0) || elem > static_cast<P>(1.0);
  });
  for (int i : out_of_range)
  {
    legendre.update_row(
        i, std::vector<P>(std::max(degree, 1), static_cast<P>(0.0)));
    legendre_prime.update_row(
        i, std::vector<P>(std::max(degree, 1), static_cast<P>(0.0)));
  }

  if (degree > 0)
  {
    // "scaling to use normalization
    legendre       = legendre * static_cast<P>(std::sqrt(2.0));
    legendre_prime = legendre_prime * static_cast<P>(std::sqrt(2.0));
  }
  return {legendre, legendre_prime};
}

// From the matlab:

//% lgwt.m
//% This script is for computing the Legendre-Gauss nodes (roots on x) and
// weights % on an interval % [lower_bound,upper_bound] for Legendre
// polynomial degree num_points % These are later used for computing
// definite integrals using Legendre-Gauss % Quadrature.
//%
//% Suppose you have a continuous function f(x) which is defined on
//% [lower_bound,upper_bound]
//% which you can evaluate at any x in [lower_bound, upper_bound].
//% Simply evaluate it at all of
//% the values contained in the x vector to obtain a vector f. Then compute
//% the definite integral using sum(f.*w);
//%
//% Written by Greg von Winckel - 02/25/2004

// uses default quadrature number of max(10, degree+1)
// unless use_degree_points is set

// return[0] are the x_roots, return[1] are the weights
template<typename P>
std::array<fk::vector<P>, 2>
legendre_weights(int const degree, P const lower_bound, P const upper_bound,
                 bool const use_degree_points)
{
  expect(degree > 0);
  expect(lower_bound < upper_bound);

  auto const default_quad_number = [](int const degree) {
    static int constexpr minimum_quadrature = 10;
    return std::max(minimum_quadrature, degree + 1);
  };

  int const num_points =
      use_degree_points ? degree : default_quad_number(degree);

  // prepare output vectors
  // the number of roots of a Legendre polynomial is equal to its degree
  fk::vector<P> x_roots(num_points);
  fk::vector<P> weights(num_points);

  // x_linspace=linspace(-1,1,num_points)';
  // This is a linearly spaced vector used to compose the initial guess
  // of the roots x_roots done next
  fk::vector<P> const x_linspace =
      linspace(static_cast<P>(-1.0), static_cast<P>(1.0), num_points);

  // Initial guess at the roots for the Legendre polynomial of degree num_points

  // x_roots = cos((2*(0:num_points-1)'+1)*pi /
  // (2*(num_points-1)+2))+(0.27/num_points) *
  // sin(pi*x_linspace*((num_points-1)/(num_points+1);

  // It is unkown where this guess comes from, but seems to work well

  // The above operation is split into two pieces, the cos term (performed on
  // x_roots) and then the sin term (performed on x_roots2) then they are added
  // together
  x_roots = linspace(static_cast<P>(0.0), static_cast<P>((num_points - 1)),
                     num_points);
  std::transform(x_roots.begin(), x_roots.end(), x_roots.begin(), [&](P &elem) {
    return std::cos((2 * elem + 1) * M_PI /
                    static_cast<P>((2 * (num_points - 1) + 2)));
  });

  fk::vector<P> x_roots2(x_linspace);
  std::transform(
      x_roots2.begin(), x_roots2.end(), x_roots2.begin(), [&](P &elem) {
        return (static_cast<P>(0.27) / num_points) *
               std::sin(M_PI * elem * (num_points - 1) / (num_points + 1));
      });

  x_roots = x_roots + x_roots2;

  //% Legendre-Gauss Vandermonde Matrix
  //% Legendre polynomial values for poly degree 0 to num_points_plus_one
  //% at the (estimated) zeros of the polynomial of degree num_points
  // legendre_y_values=zeros(num_points,num_points+2);
  fk::matrix<P> legendre_y_values(num_points, (num_points + 1));
  // The y values of the derivative of the legendre polynomial
  // of degree equals num_points at each of the (estimated) root
  // locations
  fk::vector<P> legendre_prime_y_values(num_points);

  // Set x_roots_initial to a value that will fail the
  // closeness comparison for the Newton iteration
  // x_roots_initial=2
  fk::vector<P> x_roots_initial(num_points);
  std::fill(x_roots_initial.begin(), x_roots_initial.end(),
            static_cast<P>(2.0));
  P const eps = std::numeric_limits<P>::epsilon();

  // This piece of the code uses Newton's method to solve for the
  // Legendre polynomial roots
  // x_roots = x_roots_initial - f(x_roots_initial)/f'(x_roots_initial)
  // where the function f is the legendre polynomial of degree num_points
  //% Iterate until new points are uniformly within epsilon of old points
  // Recursion formulae for Legendre polynomials as wellas the derivative
  // are used
  // while max(abs(x_roots-x_roots_initial))>eps
  fk::vector<P> diff(num_points);
  auto const abs_diff = [&](P const &y_elem, P const &y0_elem) {
    return std::fabs(y_elem - y0_elem);
  };
  std::transform(x_roots.begin(), x_roots.end(), x_roots_initial.begin(),
                 diff.begin(), abs_diff);

  while (*std::max_element(diff.begin(), diff.end()) > eps)
  {
    // Set values of Legendre polynomial P0 = 1;
    // legendre_y_values column 0
    legendre_y_values.update_col(
        0, std::vector<P>(legendre_y_values.nrows(), static_cast<P>(1.0)));

    // Set values of Legendre polynomial P1 = x_roots;
    // legendre_y_values column 1
    legendre_y_values.update_col(1, x_roots);

    // for i=1:num_points-1
    // Set values of Legendre polynomial P_i
    // we set the i+1th column of L at each iter
    for (int i = 1; i < (num_points); ++i)
    {
      // Legendre polynomials P_i-1 and P_i are used in a recurrence relation to
      // produce P_i+1
      // prev is P_i-1(x_roots) and current is P_i(x_roots)
      // next is P_i+1(x_roots)
      fk::vector<P> const prev = legendre_y_values.extract_submatrix(
          0, i - 1, legendre_y_values.nrows(), 1);
      fk::vector<P> const current = legendre_y_values.extract_submatrix(
          0, i, legendre_y_values.nrows(), 1);

      fk::vector<P> next(current.size());

      // this loop for setting the next column is a little obscure, but doing
      // step by step with transforms was prohibitively slow when we invoke this
      // function from multiwavelet gen.
      // P_i+1(x_roots) = ((2*i+1)*x_roots*P_i(x_roots) -
      // i*P_i-1(x_roots))/(i+1)
      P scale = (static_cast<P>(2.0 * i) + 1);
      for (int j = 0; j < next.size(); ++j)
      {
        next(j) = ((x_roots(j) * scale * current(j)) - (prev(j) * i)) /
                  static_cast<P>(i + 1);
      }
      legendre_y_values.update_col(i + 1, next);
    }
    // P'_i(x_roots) = i*(P_i-1(x_roots) -
    // x_roots*P_i(x_roots))/(1-x_roots.^2) Here we want to produce
    // P'_num_points(x_roots) so P_num_points and
    // P_num_points-1 are needed
    fk::vector<P> legendre_num_points = legendre_y_values.extract_submatrix(
        0, num_points, legendre_y_values.nrows(), 1);
    fk::vector<P> legendre_num_points_minus_one =
        legendre_y_values.extract_submatrix(0, num_points - 1,
                                            legendre_y_values.nrows(), 1);
    // legendre_polynomial_scaled = P_i(x_roots)*x_roots
    fk::vector<P> legendre_num_points_scaled(legendre_num_points.size());
    std::transform(legendre_num_points.begin(), legendre_num_points.end(),
                   x_roots.begin(), legendre_num_points_scaled.begin(),
                   std::multiplies<P>());
    // x_roots_denominator = (1-x_roots^2)
    fk::vector<P> const x_roots_denominator = [&] {
      fk::vector<P> copy_y(x_roots.size());
      std::transform(
          x_roots.begin(), x_roots.end(), x_roots.begin(), copy_y.begin(),
          [](P &y, P &y_same) { return static_cast<P>(1.0) - y * y_same; });
      return copy_y;
    }();
    // calculation of part of legendre_prime_y_values
    // P'i(x_roots) = ((i+1)*x_roots*P_i(x_roots) - (i+1)*P_i+1(x_roots))
    // division by (1-x_roots^2) next
    legendre_prime_y_values =
        (legendre_num_points_minus_one - legendre_num_points_scaled) *
        (num_points);
    auto const element_division = [](P const &one, P const &two) {
      return one / two;
    };
    // Finishes calculation for legendre_prime_y_values
    // P'i(x_roots) = ((i+1)*x_roots*P_i(x_roots) - (i+1)*P_i+1(x_roots))
    // division by (1-x_roots^2)
    std::transform(legendre_prime_y_values.begin(),
                   legendre_prime_y_values.end(), x_roots_denominator.begin(),
                   legendre_prime_y_values.begin(), element_division);

    x_roots_initial = x_roots;

    // x_roots=x_roots_initial-legendre_num_points./legendre_prime_y_values;
    std::transform(legendre_num_points.begin(), legendre_num_points.end(),
                   legendre_prime_y_values.begin(), legendre_num_points.begin(),
                   element_division);
    x_roots = x_roots_initial - legendre_num_points;

    // diff = abs(x_roots-x_roots_initial)
    std::transform(x_roots.begin(), x_roots.end(), x_roots_initial.begin(),
                   diff.begin(), abs_diff);
  }

  //% Compute the weights
  // w=(upper_bound-lower_bound)./((1-x_roots^2).*legendre_prime_y_values.^2);
  std::transform(
      x_roots.begin(), x_roots.end(), legendre_prime_y_values.begin(),
      weights.begin(), [&](P &y_elem, P &lp_elem) {
        return (upper_bound - lower_bound) /
               ((static_cast<P>(1.0) - y_elem * y_elem) * lp_elem * lp_elem);
      });

  //% Linear map from[-1,1] to [lower_bound,upper_bound]
  // x_roots=(lower_bound*(1-x_roots)+upper_bound*(1+x_roots))/2;
  std::transform(x_roots.begin(), x_roots.end(), x_roots.begin(), [&](P &elem) {
    return (lower_bound * (1 - elem) + upper_bound * (1 + elem)) / 2;
  });

  // x=x(end:-1:1);
  // w=w(end:-1:1);
  std::reverse(x_roots.begin(), x_roots.end());
  std::reverse(weights.begin(), weights.end());

  return std::array<fk::vector<P>, 2>{x_roots, weights};
}

// explicit instatiations
#define X(T)                                        \
  template std::array<fk::matrix<T>, 2> legendre(   \
      fk::vector<T> const domain, int const degree, \
      legendre_normalization const norm);
#include "type_list_float.inc"
#undef X

#define X(T)                                                      \
  template std::array<fk::vector<T>, 2> legendre_weights(         \
      int const degree, T const lower_bound, T const upper_bound, \
      bool const use_degree_points);
#include "type_list_float.inc"
#undef X
