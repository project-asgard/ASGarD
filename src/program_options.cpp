#include "program_options.hpp"
#include "build_info.hpp"
#include "clara.hpp"
#include "distribution.hpp"
#include <iostream>

parser::parser(int argc, char **argv)
{
  bool show_help = false;

  // Parsing...
  auto cli =
      clara::detail::Help(show_help) |
      clara::detail::Opt(cfl, "cfl")["-c"]["--cfl"](
          "The Courant-Friedrichs-Lewy (CFL) condition") |
      clara::detail::Opt(dt, "dt")["-t"]["--dt"]("Size of time steps") |
      clara::detail::Opt(degree, "degree")["-d"]["--degree"](
          "Terms in legendre basis polynomials") |
      clara::detail::Opt(use_full_grid)["-f"]["--full_grid"](
          "Use full grid (vs. sparse grid)") |
      clara::detail::Opt(use_implicit_stepping)["-i"]["--implicit"](
          "Use implicit time advance (vs. explicit)") |
      clara::detail::Opt(solver_str, "solver_str")["-s"]["--solver"](
          "Solver to use (direct or gmres) for implicit advance") |
      clara::detail::Opt(level, "level")["-l"]["--level"](
          "Stating hierarchical levels (resolution)") |
      clara::detail::Opt(max_level, "max level")["-m"]["--max_level"](
          "Maximum hierarchical levels (resolution) for adaptivity") |
      clara::detail::Opt(num_time_steps, "time steps")["-n"]["--num_steps"](
          "Number of iterations") |
      clara::detail::Opt(pde_str, "pde_str")["-p"]["--pde"](
          "PDE to solve; see options.hpp for list") |
      clara::detail::Opt(do_poisson)["-e"]["--electric_solve"](
          "Do poisson solve for electric field") |
      clara::detail::Opt(wavelet_output_freq,
                         "wavelet_output_freq")["-w"]["--wave_freq"](
          "Frequency in steps for writing wavelet space "
          "output") |
      clara::detail::Opt(realspace_output_freq,
                         "realspace_output_freq")["-r"]["--real_freq"](
          "Frequency in steps for writing realspace output");

  auto result = cli.parse(clara::detail::Args(argc, argv));
  if (!result)
  {
    std::cerr << "Error in command line parsing: " << result.errorMessage()
              << '\n';
    valid = false;
  }
  if (show_help)
  {
    std::cerr << cli << '\n';
    exit(0);
  }

  // Validation...
  if (cfl != NO_USER_VALUE_FP)
  {
    if (cfl <= 0.0)
    {
      std::cerr << "CFL must be positive" << '\n';
      valid = false;
    }
    if (dt != NO_USER_VALUE_FP)
    {
      std::cerr << "CFL and explicit dt options are mutually exclusive" << '\n';
      valid = false;
    }
  }
  else
  {
    cfl = DEFAULT_CFL;
  }

  if (degree < 1 && degree != NO_USER_VALUE)
  {
    std::cerr << "Degree must be a natural number" << '\n';
    valid = false;
  }
  if (level < 2 && level != NO_USER_VALUE)
  {
    std::cerr << "Level must be greater than one" << '\n';
    valid = false;
  }
  if (max_level < level)
  {
    std::cerr << "Maximum level must be greater than starting level" << '\n';
    valid = false;
  }
  if (dt != NO_USER_VALUE_FP && dt <= 0.0)
  {
    std::cerr << "Provided dt must be positive" << '\n';
    valid = false;
  }
  if (num_time_steps < 1)
  {
    std::cerr << "Number of timesteps must be a natural number" << '\n';
    valid = false;
  }

  auto const choice = pde_mapping.find(pde_str);
  if (choice == pde_mapping.end())
  {
    std::cerr << "Invalid pde choice; see options.hpp for valid choices"
              << '\n';
    valid = false;
  }
  else
  {
    pde_choice = pde_mapping.at(pde_str);
  }

  if (realspace_output_freq < 0 || wavelet_output_freq < 0)
  {
    std::cerr << "Write frequencies must be non-negative" << '\n';
    valid = false;
  }

  if (realspace_output_freq > num_time_steps ||
      wavelet_output_freq > num_time_steps)
  {
    std::cerr << "Requested a write frequency > number of steps - no output "
                 "will be produced"
              << '\n';
    valid = false;
  }

#if !defined(ASGARD_IO_HIGHFIVE) && !defined(ASGARD_IO_MATLAB_DIR)
  if (realspace_output_freq > 0 || wavelet_output_freq > 0)
  {
    std::cerr << "Must build with ASGARD_IO_HIGHFIVE or ASGARD_IO_MATLAB_DIR "
                 "to write output"
              << '\n';
    valid = false;
  }
#endif

  if (use_implicit_stepping)
  {
    if (solver_str == "none")
    {
      solver_str = "direct";
    }
    auto const choice = solver_mapping.find(solver_str);
    if (choice == solver_mapping.end())
    {
      std::cerr << "Invalid solver choice; see options.hpp for valid choices\n";
      valid = false;
    }
    else
    {
      solver = solver_mapping.at(solver_str);
    }
  }
  else // explicit time advance
  {
    if (solver_str != "none")
    {
      std::cerr << "Must set implicit (-i) flag to select a solver\n";
      valid = false;
    }
  }

#ifdef ASGARD_USE_CUDA
  if (use_implicit_stepping)
  {
    std::cerr << "GPU acceleration not implemented for implicit stepping\n";
    valid = false;
  }
#endif

#ifdef ASGARD_USE_MPI
  if (use_implicit_stepping && get_num_ranks() > 1)
  {
    std::cerr << "Distribution not implemented for implicit stepping\n";
    valid = false;
  }
  if (realspace_output_freq > 0)
  {
    std::cerr << "Distribution does not yet support realspace transform\n";
    valid = false;
  }
#endif
}

bool parser::using_implicit() const { return use_implicit_stepping; }
bool parser::using_full_grid() const { return use_full_grid; }
bool parser::do_poisson_solve() const { return do_poisson; }

int parser::get_level() const { return level; }
int parser::get_degree() const { return degree; }
int parser::get_max_level() const { return max_level; }
int parser::get_time_steps() const { return num_time_steps; }
int parser::get_wavelet_output_freq() const { return wavelet_output_freq; }
int parser::get_realspace_output_freq() const { return realspace_output_freq; }

double parser::get_cfl() const { return cfl; }
double parser::get_dt() const { return dt; }

std::string parser::get_pde_string() const { return pde_str; }
std::string parser::get_solver_string() const { return solver_str; }

PDE_opts parser::get_selected_pde() const { return pde_choice; }
solve_opts parser::get_selected_solver() const { return solver; }

bool parser::is_valid() const { return valid; }

bool options::should_output_wavelet(int const i) const
{
  return write_at_step(i, wavelet_output_freq);
}

bool options::should_output_realspace(int const i) const
{
  return write_at_step(i, realspace_output_freq);
}

bool options::write_at_step(int const i, int const freq) const
{
  assert(i >= 0);
  assert(freq >= 0);

  if (freq == 0)
  {
    return false;
  }
  if (freq == 1)
  {
    return true;
  }
  if ((i + 1) % freq == 0)
  {
    return true;
  }
  return false;
}
