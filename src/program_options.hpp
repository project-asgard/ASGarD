#pragma once

#include "tensors.hpp"
#include "tools.hpp"

#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// implemented solvers for implicit stepping
enum class solve_opts
{
  direct,
  gmres
};

// map those choices to selection strings
using solve_map_t                       = std::map<std::string, solve_opts>;
static solve_map_t const solver_mapping = {
    {"direct", solve_opts::direct},
    {"gmres", solve_opts::gmres},
};

// the choices for supported PDE types
enum class PDE_opts
{
  continuity_1,
  continuity_2,
  continuity_3,
  continuity_6,
  fokkerplanck_1d_4p1a,
  fokkerplanck_1d_4p2,
  fokkerplanck_1d_4p3,
  fokkerplanck_1d_4p4,
  fokkerplanck_1d_4p5,
  fokkerplanck_2d_complete,
  diffusion_1,
  diffusion_2,
  // FIXME will need to add the user supplied PDE choice
};

class PDE_descriptor
{
public:
  PDE_descriptor(std::string const info, PDE_opts const pde_choice)
      : info(info), pde_choice(pde_choice){};
  std::string const info;
  PDE_opts const pde_choice;
};

//
// map those choices to selection strings
//
using pde_map_t                    = std::map<std::string, PDE_descriptor>;
static pde_map_t const pde_mapping = {
    {"continuity_1",
     PDE_descriptor("1D test case, continuity equation: df/dt + df/dx = 0",
                    PDE_opts::continuity_1)},
    {"continuity_2", PDE_descriptor("2D test case, continuity equation: df/dt "
                                    "+ v_x * df/dx + v_y * df/dy == 0",
                                    PDE_opts::continuity_2)},
    {"continuity_3", PDE_descriptor("3D test case, continuity equation, df/dt "
                                    "+ v.grad(f)==0 where v={1,1,1}",
                                    PDE_opts::continuity_3)},
    {"continuity_6",
     PDE_descriptor("6D test case, continuity equation, df/dt + b.grad_x(f) + "
                    "a.grad_v(f)==0 where b={1,1,3}, a={4,3,2}",
                    PDE_opts::continuity_6)},
    // the following are labelled according to figure number in the runaway
    // electron paper
    {"fokkerplanck_1d_4p1a",
     PDE_descriptor(
         "1D pitch angle collisional term: df/dt == d/dz ( (1-z^2) df/dz",
         PDE_opts::fokkerplanck_1d_4p1a)},
    {"fokkerplanck_1d_4p2",
     PDE_descriptor(
         "1D pitch angle collisional term: df/dt == d/dz ( (1-z^2) df/dz",
         PDE_opts::fokkerplanck_1d_4p2)},
    {"fokkerplanck_1d_4p3",
     PDE_descriptor("Radiation damping term: df/dt == -d/dz ( z(1-z^2)f )",
                    PDE_opts::fokkerplanck_1d_4p3)},

    {"fokkerplanck_1d_4p4",
     PDE_descriptor("Evolution of f's pitch angle dependence with electric "
                    "field acceleration/collision: df/dt == -E d/dz((1-z^2) f) "
                    "+ C d/dz((1-z^2) df/dz)",
                    PDE_opts::fokkerplanck_1d_4p4)},
    {"fokkerplanck_1d_4p5",
     PDE_descriptor(
         "Same as 4p4, but with radiation damping: df/dt == -E d/dz((1-z^2) f) "
         "+ C d/dz((1-z^2) df/dz) - R d/dz(z(1-z^2) f)",
         PDE_opts::fokkerplanck_1d_4p5)},

    {"fokkerplanck_2d_complete",
     PDE_descriptor("Full PDE from the 2D runaway electron paper: d/dt f(p,z) "
                    "== -div(flux_C + flux_E + flux_R)",
                    PDE_opts::fokkerplanck_2d_complete)},
    {"diffusion_1", PDE_descriptor("1D diffusion equation: df/dt = d^2 f/dx^2",
                                   PDE_opts::diffusion_1)},
    {"diffusion_2",
     PDE_descriptor("2D (1x-1y) heat equation. df/dt = d^2 f/dx^2 + d^2 f/dy^2",
                    PDE_opts::diffusion_2)}};

// class to parse command line input
class parser
{
public:
  static auto constexpr NO_USER_VALUE     = -1;
  static auto constexpr NO_USER_VALUE_FP  = std::numeric_limits<double>::min();
  static auto constexpr NO_USER_VALUE_STR = "none";

  static auto constexpr DEFAULT_CFL          = 0.01;
  static auto constexpr DEFAULT_ADAPT_THRESH = 1e-3;
  static auto constexpr DEFAULT_MAX_LEVEL    = 8;
  static auto constexpr DEFAULT_TIME_STEPS   = 10;
  static auto constexpr DEFAULT_WRITE_FREQ   = 0;
  static auto constexpr DEFAULT_USE_IMPLICIT = false;
  static auto constexpr DEFAULT_USE_FG       = false;
  static auto constexpr DEFAULT_DO_POISSON   = false;
  static auto constexpr DEFAULT_DO_ADAPT     = false;
  static auto constexpr DEFAULT_PDE_STR      = "continuity_2";
  static auto constexpr DEFAULT_PDE_OPT      = PDE_opts::continuity_2;
  static auto constexpr DEFAULT_SOLVER       = solve_opts::direct;

  // construct from command line
  explicit parser(int argc, char **argv);

  // construct from provided values - to simplify testing
  explicit parser(PDE_opts const pde_choice, fk::vector<int> starting_levels,
                  int const degree             = NO_USER_VALUE,
                  double const cfl             = DEFAULT_CFL,
                  bool const use_full_grid     = DEFAULT_USE_FG,
                  int const max_level          = DEFAULT_MAX_LEVEL,
                  int const num_steps          = DEFAULT_TIME_STEPS,
                  bool const use_implicit      = DEFAULT_USE_IMPLICIT,
                  bool const do_adapt_levels   = DEFAULT_DO_ADAPT,
                  double const adapt_threshold = DEFAULT_ADAPT_THRESH)
      : use_implicit_stepping(use_implicit), use_full_grid(use_full_grid),
        do_adapt(do_adapt_levels), starting_levels(starting_levels),
        degree(degree), max_level(max_level), num_time_steps(num_steps),
        cfl(cfl), adapt_threshold(adapt_threshold), pde_choice(pde_choice){};

  explicit parser(std::string const &pde_choice,
                  fk::vector<int> starting_levels,
                  int const degree             = NO_USER_VALUE,
                  double const cfl             = DEFAULT_CFL,
                  bool const use_full_grid     = DEFAULT_USE_FG,
                  int const max_level          = DEFAULT_MAX_LEVEL,
                  int const num_steps          = DEFAULT_TIME_STEPS,
                  bool const use_implicit      = DEFAULT_USE_IMPLICIT,
                  bool const do_adapt_levels   = DEFAULT_DO_ADAPT,
                  double const adapt_threshold = DEFAULT_ADAPT_THRESH)
      : parser(pde_mapping.at(pde_choice).pde_choice, starting_levels, degree,
               cfl, use_full_grid, max_level, num_steps, use_implicit,
               do_adapt_levels, adapt_threshold){};

  bool using_implicit() const;
  bool using_full_grid() const;
  bool do_poisson_solve() const;
  bool do_adapt_levels() const;

  fk::vector<int> get_starting_levels() const;

  int get_degree() const;
  int get_max_level() const;
  int get_time_steps() const;

  int get_wavelet_output_freq() const;
  int get_realspace_output_freq() const;

  double get_dt() const;
  double get_cfl() const;
  double get_adapt_thresh() const;

  std::string get_pde_string() const;
  std::string get_solver_string() const;

  PDE_opts get_selected_pde() const;
  solve_opts get_selected_solver() const;

  bool is_valid() const;

private:
  void print_available_pdes()
  {
    auto const max_name_length = 25;
    std::cerr << "available pdes (select using -p)"
              << "\n\n";
    std::cerr << std::left << std::setw(max_name_length) << "Argument"
              << "Description" << '\n';
    for (auto const &[pde_name, pde_enum_val] : pde_mapping)
    {
      ignore(pde_enum_val);
      tools::expect(pde_name.size() <= max_name_length);
      std::cerr << std::left << std::setw(max_name_length) << pde_name
                << pde_mapping.at(pde_name).info << '\n';
    }
  }

  fk::vector<int> ints_from_string(std::string const &number_string)
  {
    std::stringstream number_stream{number_string};
    std::vector<int> parsed_ints;
    while (!number_stream.eof())
    {
      std::string word;
      number_stream >> word;
      int temp_int;
      if (std::stringstream(word) >> temp_int)
      {
        parsed_ints.push_back(temp_int);
      }
    }
    return fk::vector<int>(parsed_ints);
  }

  bool use_implicit_stepping =
      DEFAULT_USE_IMPLICIT;             // enable implicit(/explicit) stepping
  bool use_full_grid = DEFAULT_USE_FG;  // enable full(/sparse) grid
  bool do_poisson = DEFAULT_DO_POISSON; // do poisson solve for electric field
  bool do_adapt   = DEFAULT_DO_ADAPT;   // adapt number of basis levels

  // if none are provided, default is loaded from pde
  std::string starting_levels_str = NO_USER_VALUE_STR;
  fk::vector<int> starting_levels;

  // deg of legendre basis polys. NO_USER_VALUE
  // loads default in pde
  int degree = NO_USER_VALUE;
  // max adaptivity level for any given dimension.
  int max_level = DEFAULT_MAX_LEVEL;
  // number of time loop iterations
  int num_time_steps = DEFAULT_TIME_STEPS;
  // write wavelet space output every this many iterations
  int wavelet_output_freq = DEFAULT_WRITE_FREQ;
  // timesteps between realspace output writes to disk
  int realspace_output_freq = DEFAULT_WRITE_FREQ;
  // the Courant-Friedrichs-Lewy (CFL) condition
  double cfl = NO_USER_VALUE_FP;
  // size of time steps. double::min loads default in pde
  double dt = NO_USER_VALUE_FP;
  // relative adaptivity threshold
  // max(abs(x)) for solution vector x * adapt_thresh
  // is the threshold for refining elements
  double adapt_threshold = DEFAULT_ADAPT_THRESH;

  // default
  std::string pde_str = DEFAULT_PDE_STR;
  // pde to construct/evaluate
  PDE_opts pde_choice = DEFAULT_PDE_OPT;

  // default
  std::string solver_str = NO_USER_VALUE_STR;
  // solver to use for implicit timestepping
  solve_opts solver = DEFAULT_SOLVER;

  // is there a better (testable) way to handle invalid command-line input?
  bool valid = true;
};

// simple class to hold non-pde user options
class options
{
public:
  options(parser const &user_vals)
      : starting_levels(user_vals.get_starting_levels()),
        adapt_threshold(user_vals.get_adapt_thresh()),
        max_level(user_vals.get_max_level()),
        num_time_steps(user_vals.get_time_steps()),
        wavelet_output_freq(user_vals.get_wavelet_output_freq()),
        realspace_output_freq(user_vals.get_realspace_output_freq()),
        use_implicit_stepping(user_vals.using_implicit()),
        use_full_grid(user_vals.using_full_grid()),
        do_poisson_solve(user_vals.do_poisson_solve()),
        do_adapt_levels(user_vals.do_adapt_levels()),
        solver(user_vals.get_selected_solver()){};

  bool should_output_wavelet(int const i) const;
  bool should_output_realspace(int const i) const;

  fk::vector<int> const starting_levels;

  double const adapt_threshold;

  int const max_level;
  int const num_time_steps;
  int const wavelet_output_freq;
  int const realspace_output_freq;

  bool const use_implicit_stepping;
  bool const use_full_grid;
  bool const do_poisson_solve;
  bool const do_adapt_levels;

  solve_opts const solver;

private:
  // helper for output writing
  bool write_at_step(int const i, int const freq) const;
};
