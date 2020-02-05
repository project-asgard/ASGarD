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

#include "../tensors.hpp"
#include "pde_base.hpp"

// ---------------------------------------------------------------------------
//
// the "fokkerplanck 1d - problem 4.1" pde
//
// 1D pitch angle collisional term
// df/dt == d/dz ( (1-z^2) df/dz )
//
// Here we use LDG for this second order system. We impose homogeneous
// Neumann BCs on f
//
// d/dz( (1-z^2) df/dz ) becomes
//
// d/dz (1-z^2)*q  with free (homogeneous Neumann BC)
//
// and the flux is
//
// q=df/fz  with homogeneous Dirichlet BC
//
// run parameters to match run-away electron paper
// ./asgard -p fokkerplanck_1d_4p1b -n 4801 -f -d 4 -l 5 -i -r 800
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_fokkerplanck_1d_4p1b : public PDE<P>
{
public:
  PDE_fokkerplanck_1d_4p1b(int const num_levels = -1, int const degree = -1)
      : PDE<P>(num_levels, degree, num_dims_, num_sources_, num_terms_,
               dimensions_, terms_, sources_, exact_vector_funcs_,
               exact_scalar_func_, get_dt_, do_poisson_solve_,
               has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 0;
  static int constexpr num_terms_          = 1;
  static bool constexpr do_poisson_solve_  = false;
  static bool constexpr has_analytic_soln_ = true;

  //
  // function definitions needed to build up the "dimension", "term", and
  // "source" member objects below for this PDE
  //

  // specify initial condition vector functions...
  static fk::vector<P>
  initial_condition_dim0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    auto f = analytic_solution_dim0(x, 0);
    return f;
  }

  // analytic solution

  static P phi(P const z, P const t) { return std::tanh(std::atanh(z) - t); }
  static P f0(P const z) { return std::exp(-std::pow(z,2)/0.01); }

  static fk::vector<P>
  analytic_solution_dim0(fk::vector<P> const z, P const t = 0)
  {
    fk::vector<P> f(z.size());
    for (int i = 0; i < z.size(); ++i)
    {
      auto p  = phi(z(i), t);
      auto t1 = 1 - std::pow(p, 2);
      auto t2 = 1 - std::pow(z(i), 2);
      auto t3 = f0(p);
      f(i)    = t1 / t2 * t3;
    }
    return f;
  }

  static P analytic_solution_time(P const time)
  {
    ignore(time);
    return 1.0;
  }

  // specify source functions...

  // N/A

  // get time step (dt)

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dx      = x_range / std::pow(2, dim.get_level());
    P const dt      = dx;
    // this will be scaled by CFL
    // from command line
    return dt;
  }

  // g-funcs
  static P g_func_0(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return -1.0;
  }
  static P g_func_1(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(time);
    return -1 * (1 - std::pow(x, 2));
  }
  static P g_func_2(P const x, P const time)
  {
    // suppress compiler warnings
    ignore(x);
    ignore(time);
    return 1.0;
  }

  // define dimensions
  inline static dimension<P> const dim0_ =
      dimension<P>(-1.0,                   // domain min
                   1.0,                    // domain max
                   2,                      // levels
                   2,                      // degree
                   initial_condition_dim0, // initial condition
                   "x");                   // name

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_};

  // define terms (1 in this case)
  //
  //  -d/dz ( (1-z^2)*f )
  //
  // term2_z.type = 'grad'; % grad (see coeff_matrix.m for available types)
  // term2_z.G = @(z,p,t,dat) -1.*(1-z.^2); % G function for use in coeff_matrix
  // construction. term2_z.LF = -1; % Upwind term2_z.name = 'd_dz';

  inline static partial_term<P> const partial_term_0 =
      partial_term<P>(coefficient_type::grad, g_func_1, flux_type::central,
                      boundary_condition::neumann, boundary_condition::neumann);

  inline static term<P> const term0_dim0_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "d_dx",          // name
              dim0_,           // owning dim
              {partial_term_0});

  inline static std::vector<term<P>> const terms0_ = {term0_dim0_};

  inline static term_set<P> const terms_ = {terms0_};

  // define sources

  inline static std::vector<source<P>> const sources_ = {};

  // define exact soln functions
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      analytic_solution_dim0};

  inline static scalar_func<P> const exact_scalar_func_ =
      analytic_solution_time;
};
