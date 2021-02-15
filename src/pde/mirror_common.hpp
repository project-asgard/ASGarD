#pragma once

#include "build_info.hpp"
#include "pde_base.hpp"

#include "../tensors.hpp"

#include <math.h>

namespace mirror
{
template<typename P>
struct parameters
{
  // _M_E in matlab - _M_E is the constant for e in C++
  static auto constexpr _M_E = 9.109 * 1e-31;  // electron mass in kg
  static auto constexpr M_D  = 3.3443 * 1e-27; // deuterium mass in kgs
  static auto constexpr M_H  = 1.6726 * 1e-27; // hydrogen mass in kgs
  static auto constexpr K_B =
      1.380 * 1e-23; // Boltzmann constant in Joules/Kelvin
  static auto constexpr E = 1.602 * 1e-19; // charge in Coulombs
  static auto constexpr EPS_0 =
      8.85 * 1e-12; // permittivity of free space in Farad/m
  static auto constexpr MU_0 =
      4 * M_PI * 1e-7;               // magnetic permeability in Henries/m
  static auto constexpr R_MAG = 0.4; // radius of individual loop in meters

  static auto constexpr LN_DELT = 10; // Coulomb logarithm
  static auto constexpr I_MAG =
      10; // current going through individual loop in meters
  static auto constexpr N_TURNS = 5;   // number of turns in the mirror
  static auto constexpr V_TEST  = 500; // test value for velocity in coefficient

  static auto constexpr B_0 =
      MU_0 * N_TURNS * I_MAG / (2 * R_MAG); // magnetic field under current loop

  static auto constexpr Z_TEST =
      M_PI / 2 - 1e-6; // test value for pitch angle in coefficient

  static auto constexpr V_TH = [](P const T_eV, P const m) {
    return std::sqrt(2 * T_eV * E / m);
  };

  // species b: electrons in background

  struct species_b_t
  {
    static auto constexpr N    = 4e19;
    static auto constexpr T_EV = 4;
    static auto constexpr Z    = -1;
    static auto constexpr M    = _M_E;
    static auto constexpr V_TH = []() { return parameters::V_TH(T_EV, M); };
  };
  inline static species_b_t const species_b;

  // species b2: deuterium in background
  struct species_b2_t
  {
    static auto constexpr N    = 4e19;
    static auto constexpr T_EV = 4;
    static auto constexpr Z    = 1;
    static auto constexpr M    = M_D;
    static auto constexpr V_TH = []() { return parameters::V_TH(T_EV, M); };
  };
  inline static species_b2_t const species_b2;

  // species a: electrons in beam
  struct species_a_t
  {
    static auto constexpr N    = 4e19;
    static auto constexpr T_EV = 1e3;
    static auto constexpr Z    = -1;
    static auto constexpr M    = _M_E;
    static auto constexpr V_TH = []() { return parameters::V_TH(T_EV, M); };
  };
  inline static species_a_t const species_a;

  // normalized velocity to thermal velocity
  static auto constexpr X = [](fk::vector<P> const &v, P const v_th) {
    fk::vector<P> fx(v.size());
    std::transform(v.begin(), v.end(), fx.begin(),
                   [v_th](P const x) { return x / v_th; });
    return fx;
  };

  static auto constexpr XI = [](P const s, P const r_mag) { return s / r_mag; };

  static auto constexpr PHI = [](P const x) { return std::erf(x); };

  static auto constexpr PHI_F = [](fk::vector<P> const &x) {
    fk::vector<P> fx(x.size());

    std::transform(x.begin(), x.end(), fx.begin(), [](P const val) {
      return (val + (1.0 / (2.0 * val))) * std::erf(val) +
             std::exp(-std::pow(val, 2)) / std::sqrt(M_PI);
    });
    return fx;
  };

  static auto constexpr PSI = [](fk::vector<P> const &x) {
    fk::vector<P> fx(x.size());

    std::transform(x.begin(), x.end(), fx.begin(), [](P const val) {
      if (std::abs(val) < 1e-5)
      {
        return 0.0;
      }
      auto const dphi_dx = 2.0 / std::sqrt(M_PI) * std::exp(-std::pow(val, 2));
      auto const f_val =
          1.0 / (2.0 * std::pow(val, 2)) * (PHI(val) - val * dphi_dx);
      return f_val;
    });
    return fx;
  };

  // scaling coefficient
  static auto constexpr NU_AB_0 = []() {
    auto const num = species_b.N * std::pow(E, 4) * std::pow(species_a.Z, 2) *
                     std::pow(species_b.Z, 2) * LN_DELT;
    auto const denom = 2.0 * M_PI * std::pow(EPS_0, 2) *
                       std::pow(species_a.M, 2) * std::pow(species_b.V_TH(), 3);
    return num / denom;
  };

  // slowing down frequency
  static auto constexpr NU_S = [](fk::vector<P> const &x) {
    auto num = PSI(X(x, species_b.V_TH()));
    for (auto &val : num)
    {
      val = val * NU_AB_0() * (1 + species_a.M / species_b.M);
    }
    auto const denom = X(x, species_b.V_TH());
    fk::vector<P> fx(x.size());
    for (auto i = 0; i < fx.size(); ++i)
    {
      if (denom(i) == 0.0)
        continue;
      fx(i) = num(i) / denom(i);
    }
    return fx;
  };

  // parallel diffusion frequency
  static auto constexpr NU_PAR = [](fk::vector<P> const &x) {
    auto const num = PSI(X(x, species_b.V_TH())) * NU_AB_0();
    auto denom     = X(x, species_b.V_TH());
    for (auto &elem : denom)
    {
      elem = std::pow(elem, 3);
    }
    fk::vector<P> fx(x.size());
    for (auto i = 0; i < fx.size(); ++i)
    {
      if (denom(i) == 0.0)
        continue;
      fx(i) = num(i) / denom(i);
    }
    return fx;
  };

  // deflection frequency in s^-1
  static auto constexpr NU_D = [](fk::vector<P> const &x) {
    auto const num =
        (PHI_F(X(x, species_b.V_TH())) - PSI(X(x, species_b.V_TH()))) *
        NU_AB_0();
    auto denom = X(x, species_b.V_TH());
    for (auto &elem : denom)
    {
      elem = std::pow(elem, 3);
    }
    fk::vector<P> fx(x.size());
    for (auto i = 0; i < fx.size(); ++i)
    {
      if (denom(i) == 0.0)
        continue;
      fx(i) = num(i) / denom(i);
    }
    return fx;
  };

  static auto constexpr MAXWELL = [](fk::vector<P> const &x, P const offset,
                                     P const v_th) {
    fk::vector<P> fx(x);
    for (auto &elem : fx)
    {
      elem = species_a.N / ((std::pow(M_PI, 3) / 2.0) * std::pow(v_th, 3)) *
             std::exp(-std::pow((elem - offset) / v_th, 2));
    }
    return fx;
  };

  static auto constexpr B_FUNC = [](P const s) { return std::exp(s); };

  static auto constexpr DB_DS = [](P const s) { return std::exp(s); };

  static fk::vector<P> initial_condition_v(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    auto const offset = 1e6;
    auto const vth    = 2e5;
    return MAXWELL(x, offset, vth);
  }

  static fk::vector<P> initial_condition_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x);
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const val) { return std::cos(val / 2); });
    return fx;
  }

  static fk::vector<P> initial_condition_s(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x);
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const val) { return std::exp(val); });
    return fx;
  }

  static fk::vector<P> bc_func_0(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    return fk::vector<P>(x.size());
  }

  // exp(-nu_D(v,a,b).*t);
  static fk::vector<P> bc_func_v(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    auto const offset = 0;
    auto const v_th   = species_b.V_TH();
    return MAXWELL(x, offset, v_th);
  }

  static fk::vector<P> bc_func_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x);
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const val) { return std::cos(val); });
    return fx;
  }

  static fk::vector<P> bc_func_s(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x);
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const val) { return std::exp(val); });
    return fx;
  }

  static P bc_time(P const time)
  {
    ignore(time);
    return static_cast<P>(1.0);
  }

  // source pitch function for mirror3
  static fk::vector<P> source_func_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x);
    std::transform(x.begin(), x.end(), fx.begin(), [](P const val) {
      return static_cast<P>(-0.5) *
             (std::sin(val) + 1.0 / std::tan(val) * std::cos(val));
    });
    return fx;
  }

  static fk::vector<P> source_func_v(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    return NU_D(x);
  }

  static fk::vector<P> source_func_s(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x.size());
    std::fill(fx.begin(), fx.end(), -1.0);
    return fx;
  }

  static P source_time(P const time)
  {
    ignore(time);
    return static_cast<P>(1.0);
  }

  inline static std::vector<source<P>> const sources_3D = {
      source_func_z, source_func_v, source_func_s};

  static fk::vector<P> exact_solution_z(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x);
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const val) { return std::cos(val / 2.0); });
    return fx;
  }

  static fk::vector<P> exact_solution_v(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x);
    std::transform(x.begin(), x.end(), fx.begin(), [](P const val) {
      return species_a.N /
             (std::pow(M_PI, 3) / 2.0 *
              std::pow(V_TH(species_b.T_EV, species_a.M), 3)) *
             std::exp(-std::pow(val / V_TH(species_b.T_EV, species_a.M), 2));
    });
    return fx;
  }

  static fk::vector<P> exact_solution_s(fk::vector<P> const x, P const t = 0)
  {
    ignore(t);
    fk::vector<P> fx(x);
    std::transform(x.begin(), x.end(), fx.begin(),
                   [](P const val) { return std::exp(val); });
    return fx;
  }

}; // struct parameters

} // namespace mirror
