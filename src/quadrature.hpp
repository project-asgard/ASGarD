#pragma once

#include "tensors.hpp"
#include <array>

enum legendre_normalization
{
  unnormalized,
  lin,
  matlab
};

template<typename P>
std::enable_if_t<std::is_floating_point<P>::value, std::array<fk::matrix<P>, 2>>
legendre(fk::vector<P> const domain, int const degree,
         legendre_normalization const norm = legendre_normalization::lin);

// return[0] are the roots, return[1] are the weights
// num_points: number of quadrature points
// lower and upper bound are the bounds of integration

// if use_degree_points is set, use degree quadrature points
// otherwise, use max(10, degree + 1)
template<typename P>
std::array<fk::vector<P>, 2>
legendre_weights(int const degree, int const lower_bound, int const upper_bound,
                 bool const use_degree_points = false);

// suppress implicit instatiation
extern template std::array<fk::matrix<float>, 2>
legendre(fk::vector<float> const domain, int const degree,
         legendre_normalization const norm);
extern template std::array<fk::matrix<double>, 2>
legendre(fk::vector<double> const domain, int const degree,
         legendre_normalization const norm);
extern template std::array<fk::vector<double>, 2>
legendre_weights(int const degree, int const lower_bound, int const upper_bound,
                 bool const use_degree_points);
extern template std::array<fk::vector<float>, 2>
legendre_weights(int const degree, int const lower_bound, int const upper_bound,
                 bool const use_degree_points);
