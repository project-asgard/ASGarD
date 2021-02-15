#pragma once

#include <cmath>
#include <vector>

#include "mirror_common.hpp"
#include "pde_base.hpp"

#include "../tensors.hpp"

template<typename P, PDE_case_opts user_case = PDE_case_opts::case0>
class PDE_mirror_3d : public PDE<P>
{
public:
  PDE_mirror_3d(parser const &cli_input)
      : PDE<P>(cli_input, num_dims_, num_sources_, num_terms_, dimensions_,
               terms_, sources_, exact_vector_funcs_, exact_scalar_func_,
               get_dt_, do_poisson_solve_, has_analytic_soln_)
  {
    static_assert(user_case <= PDE_case_opts::case2, "unsupported case");
  }

private:
  inline static mirror::parameters<P> const common;

  static constexpr vector_func<P> get_initial_v()
  {
    if constexpr (user_case == PDE_case_opts::case0)
    {
      return [](fk::vector<P> const x, P const t) {
        (void)t;
        auto constexpr T_eV = 0.05 * common.species_b.T_EV;
        auto const offset   = 1e6;
        return common.MAXWELL(x, common.V_TH(T_eV, common.species_a.M), offset);
      };
    }
    else if constexpr (user_case == PDE_case_opts::case1)
    {
      return [](fk::vector<P> const x, P const t) {
        (void)t;
        auto const offset = 1e6;
        return common.MAXWELL(x, 0, offset);
      };
    }
    else if constexpr (user_case == PDE_case_opts::case2)
    {
      return common.initial_condition_v;
      // unreachable given static assert in constructor
    }
    else
    {
      expect(false);
      return common.initial_condition_v;
    }
  }

  // these fields used to check correctness of specification
  static auto constexpr num_dims_          = 3;
  static auto constexpr num_sources_       = 0;
  static auto constexpr num_terms_         = 5;
  static auto constexpr do_poisson_solve_  = false;
  static auto constexpr has_analytic_soln_ = true;

  // -- define dimensions --

  inline static dimension<P> const dim0_ =
      dimension<P>(0.0,             // domain min
                   5e6,             // domain max
                   2,               // levels
                   2,               // degree
                   get_initial_v(), // initial condition
                   "v");            // name

  inline static dimension<P> const dim1_ =
      dimension<P>(0.0,                        // domain min
                   M_PI * 0.5,                 // domain max
                   2,                          // levels
                   2,                          // degree
                   common.initial_condition_z, // initial condition
                   "z");                       // name

  inline static dimension<P> const dim2_ =
      dimension<P>(0.0,                        // domain min
                   5.0,                        // domain max
                   2,                          // levels
                   2,                          // degree
                   common.initial_condition_s, // initial condition
                   "s");                       // name

  // TODO not yet implemented - for moment integral calculation
  /*
  dim_v.jacobian = @(v,p,t) 2.*pi.*v.^2;
  dim_z.jacobian = @(z,p,t) sin(z);
  dim_s.jacobian = @(s,p,t) s.*0 + 1;
  */

  inline static std::vector<dimension<P>> const dimensions_ = {dim0_, dim1_,
                                                               dim2_};

  // -- define terms --

  // -- ADVECTION

  // termS1 == -vcos(z)*df/ds
  // termS1 == q(v)*r(z)*w(s)
  // q(v) == g1(v)  [mass, g1(p) = v,  BC N/A]
  // r(z) == g2(z) [mass, g2(z) = cos(z),  BC N/A]
  // w(s) == d/ds g3(s) f [grad, g3(s) = -1, BCL= D, BCR=D]
  static P g_func_0(P const x, P const time)
  {
    ignore(time);
    return -x;
  }

  inline static partial_term<P> const pterm_0_ =
      partial_term<P>(coefficient_type::mass, g_func_0);

  static P g_func_1(P const x, P const time)
  {
    ignore(time);
    return std::cos(x);
  }

  inline static partial_term<P> const pterm_1_ =
      partial_term<P>(coefficient_type::mass, g_func_1);

  static P g_func_identity(P const x, P const time)
  {
    ignore(x);
    ignore(time);
    return 1.0;
  }

  inline static std::vector<vector_func<P>> const boundary_left_ = {
      common.bc_func_0, common.bc_func_z, common.bc_func_s};

  inline static std::vector<vector_func<P>> const boundary_right_ = {
      common.bc_func_v, common.bc_func_z, common.bc_func_s};

  inline static partial_term<P> const pterm_2_ = partial_term<P>(
      coefficient_type::grad, g_func_identity, flux_type::downwind,
      boundary_condition::dirichlet, boundary_condition::dirichlet,
      homogeneity::inhomogeneous, homogeneity::inhomogeneous, boundary_left_,
      common.bc_time, boundary_right_, common.bc_time);

  inline static term<P> const term0_v_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "S1_v",          // name
              {pterm_0_});

  inline static term<P> const term0_z_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "S1_z",          // name
              {pterm_1_});

  inline static term<P> const term0_s_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "S1_s",          // name
              {pterm_2_});

  inline static std::vector<term<P>> const termsS0_ = {term0_v_, term0_z_,
                                                       term0_s_};

  // -- MASS

  // termS2 == -vcos(z)dB/ds f
  // termS1 == q(v)*r(z)*w(s)
  // q(v) == g1(v)  [mass, g1(p) = v,  BC N/A]
  // r(z) == g2(z) [mass, g2(z) = cos(z),  BC N/A]
  // w(s) == g3(s) f [mass, g3(s) = -dB/ds/B, BCL= D, BCR=D]

  static P g_func_4(P const x, P const time)
  {
    ignore(time);
    return common.DB_DS(x) / common.B_FUNC(x);
  }

  inline static partial_term<P> const pterm_4_ =
      partial_term<P>(coefficient_type::mass, g_func_4);

  inline static term<P> const termB0_v_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "S1_v",          // name
              {pterm_0_});

  inline static term<P> const termB0_z_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "S1_z",          // name
              {pterm_1_});

  inline static term<P> const termB0_s_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "S1_s",          // name
              {pterm_4_});

  inline static std::vector<term<P>> const termsS1_ = {termB0_v_, termB0_z_,
                                                       termB0_s_};

  // termC == nu_D/(2*sin(z))*d/dz sin(z)*df/dz
  //
  // becomes
  //
  // termC == g1(v) g2(z) q(z)   [mass, g1(p) = nu_D(v), g2(z) = 1/(2sin(z))  BC
  // N/A]
  //   q(z) == d/dz g3(z) r(z)   [grad, g3(z) =  sin(z), BCL=D,BCR=D]
  //   r(z) == d/dp g4(z) f(z)   [grad, g3(p) = 1,      BCL=N,BCR=N]

  static P g_func_C0(P const x, P const time)
  {
    ignore(time);
    return common.NU_D({x})(0);
  }

  inline static partial_term<P> const pterm_C0_ =
      partial_term<P>(coefficient_type::mass, g_func_C0);

  static P g_func_C1(P const x, P const time)
  {
    ignore(time);
    return static_cast<P>(1.0) / (static_cast<P>(2.0) * std::sin(x));
  }

  inline static partial_term<P> const pterm_C1_ =
      partial_term<P>(coefficient_type::mass, g_func_C1);

  static P g_func_C2(P const x, P const time)
  {
    ignore(time);
    return std::sin(x);
  }

  inline static partial_term<P> const pterm_C2_ = partial_term<P>(
      coefficient_type::grad, g_func_C2, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::neumann);

  inline static partial_term<P> const pterm_C3_ = partial_term<P>(
      coefficient_type::grad, g_func_identity, flux_type::downwind,
      boundary_condition::neumann, boundary_condition::dirichlet,
      homogeneity::inhomogeneous, homogeneity::inhomogeneous, boundary_left_,
      common.bc_time, boundary_right_, common.bc_time);

  inline static partial_term<P> const pterm_I =
      partial_term<P>(coefficient_type::mass, g_func_identity);

  inline static term<P> const termC_v_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "C_z",           // name
              {pterm_C0_});

  inline static term<P> const termC_z_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "C_v",           // name
              {pterm_C1_, pterm_C2_, pterm_C3_});

  inline static term<P> const I_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "identity",      // name
              {pterm_I});

  inline static std::vector<term<P>> const termsC_ = {termC_v_, termC_z_, I_};

  // term V1 == 1/v^2 d/dv(v^3(m_a/(m_a + m_b))nu_s f))
  // term V1 == g(v) q(v)      [mass, g(v) = 1/v^2,  BC N/A]
  // q(v) == d/dv(g2(v)f(v))   [grad, g2(v) = v^3(m_a/(m_a + m_b))nu_s, BCL= N,
  // BCR=D]

  static P g_func_V0_0(P const x, P const time)
  {
    ignore(time);
    return static_cast<P>(1.0) / std::pow(x, 2);
  }

  inline static partial_term<P> const pterm_V0_0_ =
      partial_term<P>(coefficient_type::mass, g_func_V0_0);

  static P g_func_V0_1(P const x, P const time)
  {
    ignore(time);
    return std::pow(x, 3) * common.species_a.M * common.NU_S({x})(0) /
           (common.species_a.M + common.species_b.M);
  }

  inline static partial_term<P> const pterm_V0_1_ = partial_term<P>(
      coefficient_type::grad, g_func_V0_1, flux_type::downwind,
      boundary_condition::neumann, boundary_condition::dirichlet,
      homogeneity::inhomogeneous, homogeneity::inhomogeneous, boundary_left_,
      common.bc_time, boundary_right_, common.bc_time);

  inline static term<P> const termV0_v_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "V_v0",          // name
              {pterm_V0_0_, pterm_V0_1_});

  inline static std::vector<term<P>> const termsV0_ = {termV0_v_, I_, I_};

  // term V2 == 1/v^2 d/dv(v^4*0.5*nu_par*d/dv(f))
  // term V2 == g(v) q(v)      [mass, g(v) = 1/v^2,  BC N/A]
  // q(v) == d/dv(g2(v)r(v))   [grad, g2(v) = v^4*0.5*nu_par, BCL= D, BCR=D]
  // r(v) = d/dv(g3(v)f)       [grad, g3(v) = 1, BCL=N, BCR=N]

  static P g_func_V1_0(P const x, P const time)
  {
    ignore(time);
    return static_cast<P>(1.0) / std::pow(x, 2);
  }

  inline static partial_term<P> const pterm_V1_0_ =
      partial_term<P>(coefficient_type::mass, g_func_V1_0);

  static P g_func_V1_1(P const x, P const time)
  {
    ignore(time);
    return std::pow(x, 4) * static_cast<P>(0.5) * common.NU_PAR({x})(0);
  }

  inline static partial_term<P> const pterm_V1_1_ = partial_term<P>(
      coefficient_type::grad, g_func_V1_1, flux_type::upwind,
      boundary_condition::dirichlet, boundary_condition::neumann);

  inline static partial_term<P> const pterm_V1_2_ = partial_term<P>(
      coefficient_type::grad, g_func_identity, flux_type::downwind,
      boundary_condition::neumann, boundary_condition::dirichlet,
      homogeneity::inhomogeneous, homogeneity::inhomogeneous, boundary_left_,
      common.bc_time, boundary_right_, common.bc_time);

  inline static term<P> const termV1_v_ =
      term<P>(false,           // time-dependent
              fk::vector<P>(), // additional data vector
              "V_v1",          // name
              {pterm_V1_0_, pterm_V1_1_, pterm_V1_2_});

  inline static std::vector<term<P>> const termsV1_ = {termV1_v_, I_, I_};

  inline static term_set<P> const terms_ = {termsV0_, termsV1_, termsC_,
                                            termsS0_, termsS1_};

  // -- exact sol --

  // define exact soln functions
  inline static std::vector<vector_func<P>> const exact_vector_funcs_ = {
      common.exact_solution_v, common.exact_solution_z,
      common.exact_solution_s};

  static P exact_time(P const time)
  {
    ignore(time);
    return static_cast<P>(1.0);
  }

  inline static scalar_func<P> const exact_scalar_func_ = exact_time;

  // -- sources --

  inline static std::vector<source<P>> const sources_ = {};

  // -- dt --

  static P get_dt_(dimension<P> const &dim)
  {
    P const x_range = dim.domain_max - dim.domain_min;
    P const dt      = x_range / std::pow(2, dim.get_level());
    return dt;
  }
};
