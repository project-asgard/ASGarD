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

#include "program_options.hpp"
#include "quadrature.hpp"

#include "pde/pde_advect_blob1.hpp"
#include "pde/pde_advect_blob2.hpp"
#include "pde/pde_advect_blob3.hpp"
#include "pde/pde_advect_blob4.hpp"
#include "pde/pde_base.hpp"
#include "pde/pde_continuity1.hpp"
#include "pde/pde_continuity2.hpp"
#include "pde/pde_continuity3.hpp"
#include "pde/pde_continuity6.hpp"
#include "pde/pde_diffusion1.hpp"
#include "pde/pde_diffusion2.hpp"
#include "pde/pde_fokkerplanck1_4p3.hpp"
#include "pde/pde_fokkerplanck1_4p4.hpp"
#include "pde/pde_fokkerplanck1_4p5.hpp"
#include "pde/pde_fokkerplanck1_pitch_C.hpp"
#include "pde/pde_fokkerplanck1_pitch_E.hpp"
#include "pde/pde_fokkerplanck2_complete.hpp"
#include "tensors.hpp"

//
// this file contains the PDE factory and the utilities to
// select the PDEs being made available by the included
// implementations
//

// ---------------------------------------------------------------------------
//
// A free function factory for making pdes. eventually will want to change the
// return for some of these once we implement them...
//
// ---------------------------------------------------------------------------

template<typename P>

std::unique_ptr<PDE<P>> make_PDE(parser const &cli_input)
{
  switch (cli_input.get_selected_pde())
  {
  case PDE_opts::advect_blob_1:
    return std::make_unique<PDE_advect_blob_1d<P>>(cli_input);
  case PDE_opts::advect_blob_2:
    return std::make_unique<PDE_advect_blob_2d<P>>(cli_input);
  case PDE_opts::advect_blob_3:
    return std::make_unique<PDE_advect_blob_3d<P>>(cli_input);
  case PDE_opts::advect_blob_4:
    return std::make_unique<PDE_advect_blob_4d<P>>(cli_input);
  case PDE_opts::continuity_1:
    return std::make_unique<PDE_continuity_1d<P>>(cli_input);
  case PDE_opts::continuity_2:
    return std::make_unique<PDE_continuity_2d<P>>(cli_input);
  case PDE_opts::continuity_3:
    return std::make_unique<PDE_continuity_3d<P>>(cli_input);
  case PDE_opts::continuity_6:
    return std::make_unique<PDE_continuity_6d<P>>(cli_input);
  case PDE_opts::fokkerplanck_1d_pitch_E_case1:
    return std::make_unique<
        PDE_fokkerplanck_1d_pitch_E<P, PDE_case_opts::case0>>(cli_input);
  case PDE_opts::fokkerplanck_1d_pitch_E_case2:
    return std::make_unique<
        PDE_fokkerplanck_1d_pitch_E<P, PDE_case_opts::case1>>(cli_input);
  case PDE_opts::fokkerplanck_1d_pitch_C:
    return std::make_unique<PDE_fokkerplanck_1d_pitch_C<P>>(cli_input);
  case PDE_opts::fokkerplanck_1d_4p3:
    return std::make_unique<PDE_fokkerplanck_1d_4p3<P>>(cli_input);
  case PDE_opts::fokkerplanck_1d_4p4:
    return std::make_unique<PDE_fokkerplanck_1d_4p4<P>>(cli_input);
  case PDE_opts::fokkerplanck_1d_4p5:
    return std::make_unique<PDE_fokkerplanck_1d_4p5<P>>(cli_input);
  case PDE_opts::fokkerplanck_2d_complete:
    return std::make_unique<PDE_fokkerplanck_2d_complete<P>>(cli_input);
  case PDE_opts::diffusion_1:
    return std::make_unique<PDE_diffusion_1d<P>>(cli_input);
  case PDE_opts::diffusion_2:
    return std::make_unique<PDE_diffusion_2d<P>>(cli_input);
  default:
    std::cout << "Invalid pde choice" << std::endl;
    exit(-1);
  }
}

// WARNING for tests only!
// features rely on options, parser, and PDE constructed w/ same arguments
// shim for easy PDE creation in tests
template<typename P>
std::unique_ptr<PDE<P>>
make_PDE(PDE_opts const pde_choice, fk::vector<int> levels,
         int const degree = parser::NO_USER_VALUE,
         double const cfl = parser::DEFAULT_CFL)
{
  return make_PDE<P>(parser(pde_choice, levels, degree, cfl));
}

// old tests based on uniform level need conversion
template<typename P>
std::unique_ptr<PDE<P>>
make_PDE(PDE_opts const pde_choice, int const level = parser::NO_USER_VALUE,
         int const degree = parser::NO_USER_VALUE,
         double const cfl = parser::DEFAULT_CFL)
{
  auto const levels = [level, pde_choice]() {
    if (level == parser::NO_USER_VALUE)
    {
      return fk::vector<int>();
    }

    switch (pde_choice)
    {
    case PDE_opts::continuity_1:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::continuity_2:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::continuity_3:
      return fk::vector<int>(std::vector<int>(3, level));

    case PDE_opts::continuity_6:
      return fk::vector<int>(std::vector<int>(6, level));

    case PDE_opts::fokkerplanck_1d_pitch_E_case1:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_pitch_E_case2:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_pitch_C:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_4p3:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_4p4:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_1d_4p5:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::fokkerplanck_2d_complete:
      return fk::vector<int>(std::vector<int>(2, level));

    case PDE_opts::diffusion_1:
      return fk::vector<int>(std::vector<int>(1, level));

    case PDE_opts::diffusion_2:
      return fk::vector<int>(std::vector<int>(2, level));

    default:
      std::cout << "Invalid pde choice" << std::endl;
      exit(-1);
    }
  }();

  return make_PDE<P>(parser(pde_choice, levels, degree, cfl));
}
