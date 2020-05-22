#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "basis.hpp"
#include "fast_math.hpp"
#include "matlab_utilities.hpp"
#include "tensors.hpp"

//
// This file contains all of the interface and object definitions for our
// representation of a PDE
//

// same pi used by matlab
static double const PI = 3.141592653589793;

// for passing around vector/scalar-valued functions used by the PDE
template<typename P>
using vector_func = std::function<fk::vector<P>(fk::vector<P> const, P const)>;
template<typename P>
using scalar_func = std::function<P(P const)>;

//----------------------------------------------------------------------------
//
// Define member classes of the PDE type: dimension, term, source
//
//----------------------------------------------------------------------------

// just a small enumeration of the possibly boundary condition types needed in
// the following 'dimension' member class
enum class boundary_condition
{
  periodic,
  dirichlet,
  neumann
};

enum class homogeneity
{
  homogeneous,
  inhomogeneous
};

// helper - single element size
auto const element_segment_size = [](auto const &pde) {
  int const degree = pde.get_dimensions()[0].get_degree();
  return static_cast<int>(std::pow(degree, pde.num_dims));
};

// ---------------------------------------------------------------------------
//
// Dimension: holds all information for a single dimension in the pde
//
// ---------------------------------------------------------------------------
// forward dec
template<typename P>
class PDE;

template<typename P>
class dimension
{
public:
  P const domain_min;
  P const domain_max;
  vector_func<P> const initial_condition;
  std::string const name;
  dimension(P const domain_min, P const domain_max, int const level,
            int const degree, vector_func<P> const initial_condition,
            std::string const name)

      : domain_min(domain_min), domain_max(domain_max),
        initial_condition(initial_condition), name(name), degree_(degree)
  {
    set_level(level);
  }

  int get_level() const { return level_; }
  int get_degree() const { return degree_; }
  fk::matrix<double> const &get_to_basis_operator() const
  {
    return to_basis_operator_;
  }
  fk::matrix<double> const &get_from_basis_operator() const
  {
    return from_basis_operator_;
  }

private:
  void set_level(int level)
  {
    assert(level > 1);
    level_         = level;
    int const dofs = degree_ * fm::two_raised_to(level_);
    to_basis_operator_.clear_and_resize(dofs, dofs) =
        operator_two_scale<double>(degree_, level_);
    from_basis_operator_.clear_and_resize(dofs, dofs) =
        fk::matrix<double>(to_basis_operator_).transpose();
  }

  void set_degree(int degree)
  {
    assert(degree > 0);
    degree_        = degree;
    int const dofs = degree_ * fm::two_raised_to(level_);
    to_basis_operator_.clear_and_resize(dofs, dofs) =
        operator_two_scale<double>(degree_, level_);
    from_basis_operator_.clear_and_resize(dofs, dofs) =
        fk::matrix<double>(to_basis_operator_).transpose();
  }

  int level_;
  int degree_;
  fk::matrix<double> to_basis_operator_;
  fk::matrix<double> from_basis_operator_;

  friend class PDE<P>;
};

enum class coefficient_type
{
  grad,
  mass,
  diff
};

enum class flux_type
{

  downwind      = -1,
  central       = 0,
  upwind        = 1,
  lax_friedrich = 0
};

// ---------------------------------------------------------------------------
//
// Term: describes a single term in the pde for operator matrix
// construction
//
// ---------------------------------------------------------------------------

// FIXME need to work on relationship with dimension
// do dimensions own terms? need dimension info in
// term construction...

using g_func_type = std::function<double(double const, double const)>;

template<typename P>
class partial_term
{
public:
  static double null_gfunc(double const x, double const t)
  {
    ignore(x);
    ignore(t);
    return 1.0;
  }

  static P null_scalar_func(P const p) { return p; }

  partial_term(coefficient_type const coeff_type,
               g_func_type const g_func       = null_gfunc,
               flux_type const flux           = flux_type::central,
               boundary_condition const left  = boundary_condition::neumann,
               boundary_condition const right = boundary_condition::neumann,
               homogeneity const left_homo    = homogeneity::homogeneous,
               homogeneity const right_homo   = homogeneity::homogeneous,
               std::vector<vector_func<P>> const left_bc_funcs = {},
               scalar_func<P> const left_bc_time_func = null_scalar_func,
               std::vector<vector_func<P>> const right_bc_funcs = {},
               scalar_func<P> const right_bc_time_func = null_scalar_func)

      : coeff_type(coeff_type), g_func(g_func), flux(flux), left(left),
        right(right), left_homo(left_homo), right_homo(right_homo),
        left_bc_funcs(left_bc_funcs), right_bc_funcs(right_bc_funcs),
        left_bc_time_func(left_bc_time_func),
        right_bc_time_func(right_bc_time_func)
  {}

  P get_flux_scale() const { return static_cast<P>(flux); };

  coefficient_type const coeff_type;

  g_func_type const g_func;

  flux_type const flux;

  boundary_condition const left;

  boundary_condition const right;

  homogeneity const left_homo;
  homogeneity const right_homo;
  std::vector<vector_func<P>> const left_bc_funcs;
  std::vector<vector_func<P>> const right_bc_funcs;
  scalar_func<P> const left_bc_time_func;
  scalar_func<P> const right_bc_time_func;

  fk::matrix<P> const &get_coefficients() const { return coefficients_; }

  void set_coefficients(fk::matrix<P> const &new_coefficients)
  {
    this->coefficients_.clear_and_resize(
        new_coefficients.nrows(), new_coefficients.ncols()) = new_coefficients;
  }

private:
  fk::matrix<P> coefficients_;
};

template<typename P>
class term
{
  static P g_func_default(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

public:
  term(bool const time_dependent, fk::vector<P> const data,
       std::string const name, dimension<P> const owning_dim,
       std::initializer_list<partial_term<P>> const partial_terms)
      : time_dependent(time_dependent), name(name), owning_dim(owning_dim),
        partial_terms(partial_terms), data_(data)

  {
    set_data(owning_dim, data);
    set_coefficients(owning_dim, eye<P>(degrees_freedom(owning_dim)));
  }

  void set_data(dimension<P> const &owning_dim, fk::vector<P> const &data)
  {
    int const degrees_freedom_1d = degrees_freedom(owning_dim);
    if (data.size() != 0)
    {
      assert(data.size() == degrees_freedom_1d);
      data_ = data;
    }
    else
    {
      this->data_.resize(degrees_freedom_1d);
      this->data_ = fk::vector<P>(std::vector<P>(degrees_freedom_1d, 1.0));
    }
  }

  fk::vector<P> get_data() const { return data_; };

  void set_coefficients(dimension<P> const &owning_dim,
                        fk::matrix<P> const &new_coefficients)
  {
    int const degrees_freedom_1d = degrees_freedom(owning_dim);
    assert(degrees_freedom_1d == new_coefficients.nrows());
    assert(degrees_freedom_1d == new_coefficients.ncols());
    this->coefficients_.clear_and_resize(degrees_freedom_1d,
                                         degrees_freedom_1d) =
        new_coefficients.clone_onto_device();
  }

  void set_partial_coefficients(fk::matrix<P> const &coeffs, int const pterm)
  {
    assert(pterm >= 0);
    assert(pterm < static_cast<int>(partial_terms.size()));
    partial_terms[pterm].set_coefficients(coeffs);
  }

  fk::matrix<P, mem_type::owner, resource::device> const &
  get_coefficients() const
  {
    return coefficients_;
  }

  // small helper to return degrees of freedom given dimension
  int degrees_freedom(dimension<P> const &d) const
  {
    return d.get_degree() * static_cast<int>(std::pow(2, d.get_level()));
  }

  std::vector<partial_term<P>> const &get_partial_terms() const
  {
    return partial_terms;
  }

  // public but const data. no getters
  bool const time_dependent;
  std::string const name;
  dimension<P> const owning_dim;

private:
  std::vector<partial_term<P>> partial_terms;

  // this is to hold data that may change over the course of the simulation,
  // from any source, that is used in operator construction.
  //
  // initialized to one if not provided at instantiation, which performs an
  // identity operation where this is used, until set by outside source.
  fk::vector<P> data_;

  // scale the jump operator in coefficient construction by this amount,
  // determined by flux type. 0 or 1 for central or upwind, respectively,
  // and df/du for lax freidrich. should not be set after construction for
  // central or upwind.

  // operator matrix for this term at a single dimension
  fk::matrix<P, mem_type::owner, resource::device> coefficients_;
};

// ---------------------------------------------------------------------------
//
// Source: a pde can have arbitrarily many, given that each has dimension-many
// vector valued functions and one scalar valued function (for time)
//
// ---------------------------------------------------------------------------

template<typename P>
class source
{
public:
  source(std::vector<vector_func<P>> const source_funcs,
         scalar_func<P> const time_func)

      : source_funcs(source_funcs), time_func(time_func)
  {}

  // public but const data. no getters
  std::vector<vector_func<P>> const source_funcs;
  scalar_func<P> const time_func;
};

// ---------------------------------------------------------------------------
//
// abstract base class defining interface for PDEs
//
// ----------------------------------------------------------------------------
template<typename P>
using term_set = std::vector<std::vector<term<P>>>;
template<typename P>
using dt_func = std::function<P(dimension<P> const &dim)>;

template<typename P>
class PDE
{
public:
  PDE(int const num_levels, int const degree, int const num_dims,
      int const num_sources, int const num_terms,
      std::vector<dimension<P>> const dimensions, term_set<P> const terms,
      std::vector<source<P>> const sources,
      std::vector<vector_func<P>> const exact_vector_funcs,
      scalar_func<P> const exact_time, dt_func<P> const get_dt,
      bool const do_poisson_solve  = false,
      bool const has_analytic_soln = false);

  // public but const data.
  int const num_dims;
  int const num_sources;
  int const num_terms;

  std::vector<source<P>> const sources;
  std::vector<vector_func<P>> const exact_vector_funcs;
  scalar_func<P> const exact_time;
  bool const do_poisson_solve;
  bool const has_analytic_soln;

  virtual ~PDE() {}

  std::vector<dimension<P>> const &get_dimensions() const
  {
    return dimensions_;
  }

  term_set<P> const &get_terms() const { return terms_; }

  fk::matrix<P, mem_type::owner, resource::device> const &
  get_coefficients(int const term, int const dim) const
  {
    assert(term >= 0);
    assert(term < num_terms);
    assert(dim >= 0);
    assert(dim < num_dims);
    return terms_[term][dim].get_coefficients();
  }

  /* gives a vector of partial_term matrices to the term object so it can
     construct the full operator matrix */
  void
  set_coefficients(fk::matrix<P> const &coeffs, int const term, int const dim)
  {
    assert(term >= 0);
    assert(term < num_terms);
    assert(dim >= 0);
    assert(dim < num_dims);
    terms_[term][dim].set_coefficients(dimensions_[dim], coeffs);
  }

  void set_partial_coefficients(int const term, int const dim, int const pterm,
                                fk::matrix<P> const &coeffs)
  {
    assert(term >= 0);
    assert(term < num_terms);
    assert(dim >= 0);
    assert(dim < num_dims);
    terms_[term][dim].set_partial_coefficients(coeffs, pterm);
  }

  P get_dt() const { return dt_; };

  void regenerate_coefficients(P const time = 0.0, bool const rotate = true);

private:
  fk::matrix<double>
  generate_coefficients(dimension<P> const &dim, term<P> const &term_1D,
                        partial_term<P> const &pterm, P const time = 0.0,
                        bool const rotate = true);

  std::vector<dimension<P>> dimensions_;
  term_set<P> terms_;
  P dt_;
};
