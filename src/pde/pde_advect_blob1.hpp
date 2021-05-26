#pragma once
#include <algorithm>
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
// the "advect blob 1d" pde in order to highlight usage of adaptivity
// 1D advect a blob across a domain. Use for feature tracking testing in the
// adaptiviy.
//
// 1D test case using advect blob equation, i.e.,
// df/dt == -v * df/dx
// In matlab: asgard(@advect_blob1,'deg',5','lev',4,'num_steps',5,'dt',0.002)
// Be aware
//  DLG - analytic solution won't work past about t = 0.002*500
//
// ---------------------------------------------------------------------------
template<typename P>
class PDE_advect_blob_1d : public PDE<P>
{
public:
  PDE_advect_blob_1d(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {}

private:
  // these fields will be checked against provided functions to make sure
  // everything is specified correctly

  static int constexpr num_dims_           = 1;
  static int constexpr num_sources_        = 1;
  static int constexpr num_terms_          = 1;
  static double constexpr blob_speed_      = 2;   // v (matlab)
  static double constexpr blob_width_      = 0.1; // sig (matlab)
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
    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(),
        // a_x = @(x,p,t)
        // exp(-(x-p.v*t).^2/p.sig.^2)+exp(-(x-p.v*t+2).^2/p.sig.^2);
        [t](P const &x) {
          return exp(-pow((x - blob_speed_ * t), 2) / pow(blob_width_, 2)) +
                 exp(-pow((x - blob_speed_ * t + 2), 2) / pow(blob_width_, 2));
        });
    return fx;
  }

  // specify exact solution vectors/time function...
  static fk::vector<P> exact_solution_dim0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(
        x.begin(), x.end(), fx.begin(),
        // a_x = @(x,p,t)
        // exp(-(x-p.v*t).^2/p.sig.^2)+exp(-(x-p.v*t+2).^2/p.sig.^2);
        [t](P const &x) {
          return exp(-pow((x - blob_speed_ * t), 2) / pow(blob_width_, 2)) +
                 exp(-pow((x - blob_speed_ * t + 2), 2) / pow(blob_width_, 2));
        });
    return fx;
  }

  static P exact_time(P const time) { return 1.; }

  // specify source functions...

  // source 0
  static fk::vector<P> source_0_dim0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const &x) { return 0.; });
    return fx;
  }

  static P source_0_time(P const time) { return 0.; }

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dx      = x_range / std::pow(2, dim.get_level());
    // return dx; this will be scaled by CFL
    // from command line
    return dx;
  }
  // g-funcs for terms (optional)
  static P g_func_0(P const x, P const time)
  {
    /* From matlab
     *g1 = @(x,p,t,dat) x.*0-p.v;
     *pterm1 = GRAD(num_dims,g1,-1,'P','P');
     * */
    ignore(x);    // suppress compiler warnings
    ignore(time); // suppress compiler warnings
    return -blob_speed_;
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

  inline static const partial_term<P> partial_term_0 = partial_term<P>(
      coefficient_type::grad, g_func_0, flux_type::downwind,
      boundary_condition::periodic, boundary_condition::periodic);

  // define terms (1 in this case)
  inline static term<P> term0_dim0_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "d_dx",          // name
              {partial_term_0});

  inline static std::vector<term<P>> const terms0_ = {term0_dim0_};

  inline static term_set<P> const terms_ = {terms0_};

  inline static std::vector<source<P>> const sources_ = {};

  // define exact soln functions
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      exact_solution_dim0};

  inline static scalar_func<P> const exact_scalar_func_ = exact_time;
};
