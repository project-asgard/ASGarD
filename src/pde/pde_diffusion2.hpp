#pragma once
#include "../pde_base.hpp"

template<typename P>
class PDE_diffusion_2d : public PDE<P>
{
public:
  PDE_diffusion_2d(int const num_levels = -1, int const degree = -1)
      : PDE<P>(num_levels, degree, num_dims_, num_sources_, num_terms_,
               dimensions_, terms_, sources_, exact_vector_funcs_,
               exact_scalar_func_, get_dt_, do_poisson_solve_,
               has_analytic_soln_)
  {}

private:
  static int constexpr num_dims_           = 2;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 2;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;
  static int constexpr default_level       = 2;
  static int constexpr default_degree      = 2;
  static int constexpr domain_min          = 0;
  static int constexpr domain_max          = 1;

  static fk::vector<P>
  initial_condition_dim(fk::vector<P> const &x, P const t = 0)
  {
    static double p     = -2.0 * PI * PI;
    P const coefficient = std::exp(p * t);

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [coefficient](P const x_value) -> P {
                     return coefficient * std::cos(PI * x_value);
                   });

    return fx;
  }

  /* Define the dimension */
  inline static dimension<P> const dim_0 =
      dimension<P>(domain_min, domain_max, default_level, default_degree,
                   initial_condition_dim, "x");

  inline static dimension<P> const dim_1 =
      dimension<P>(domain_min, domain_max, default_level, default_degree,
                   initial_condition_dim, "y");

  inline static std::vector<dimension<P>> const dimensions_ = {dim_0, dim_1};

  /* build the terms */
  inline static partial_term<P> const partial_term_I_ = partial_term<P>(
      coefficient_type::mass, partial_term<P>::null_gfunc, flux_type::central,
      boundary_condition::periodic, boundary_condition::periodic);

  inline static term<P> const I_0 =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "massY",         // name
              dim_0,           // owning dim
              {partial_term_I_});

  inline static term<P> const I_1 =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "massY",         // name
              dim_1,           // owning dim
              {partial_term_I_});

  inline static const partial_term<P> partial_term_0 = partial_term<P>(
      coefficient_type::grad, partial_term<P>::null_gfunc, flux_type::upwind,
      boundary_condition::neumann, boundary_condition::neumann);

  static fk::vector<P> bc_func(fk::vector<P> const x, P const t)
  {
    ignore(t);

    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const x_value) -> P { return std::cos(PI * x_value); });

    return fx;
  }

  static P bc_time_func(P const t)
  {
    /* e^(-2 * pi^2 * t )*/
    static double const p = -2.0 * PI * PI;
    return std::exp(p * t);
  }

  inline static const partial_term<P> partial_term_1 = partial_term<P>(
      coefficient_type::grad, partial_term<P>::null_gfunc, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::inhomogeneous, homogeneity::inhomogeneous,
      {bc_func, bc_func}, bc_time_func, {bc_func, bc_func}, bc_time_func);

  inline static term<P> const term_0 =
      term<P>(true,            // time-dependent
              fk::vector<P>(), // additional data vector
              "",              // name
              dim_0,           // owning dim
              {partial_term_0, partial_term_1});

  inline static term<P> const term_1 =
      term<P>(true,            // time-dependent
              fk::vector<P>(), // additional data vector
              "",              // name
              dim_1,           // owning dim
              {partial_term_0, partial_term_1});

  inline static std::vector<term<P>> const terms_0 = {term_0, I_0};
  inline static std::vector<term<P>> const terms_1 = {I_1, term_1};

  inline static term_set<P> const terms_ = {terms_0, terms_1};

  /* exact solutions */
  static fk::vector<P> exact_solution(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return std::cos(PI * x); });
    return fx;
  }

  static P exact_time(P const time)
  {
    static double neg_two_pi_squared = -2.0 * PI * PI;

    return std::exp(neg_two_pi_squared * time);
  }

  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution, exact_solution};

  /* This is not used ever */
  static P exact_scalar_func(P const t)
  {
    static double neg_two_pi_squared = -2.0 * PI * PI;

    return std::exp(neg_two_pi_squared * t);
  }

  inline static scalar_func<P> const exact_scalar_func_ = exact_scalar_func;

  static P get_dt_(dimension<P> const &dim)
  {
    /* (1/2^level)^2 = 1/4^level */
    /* return dx; this will be scaled by CFL from command line */
    return std::pow(0.25, dim.get_level());
  }

  /* problem contains no sources */
  inline static std::vector<source<P>> const sources_ = {};
};
