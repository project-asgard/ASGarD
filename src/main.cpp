#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "tools.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>
#include <timemory/timemory.hpp>

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

int main(int argc, char **argv)
{
  // -- parse cli
  tim::timemory_init(argc, argv);
  profiling::start("Initialization");
  parser const cli_input(argc, argv);
  if (!cli_input.is_valid())
  {
    node_out() << "invalid cli string; exiting" << '\n';
    exit(-1);
  }
  options const opts(cli_input);

  // -- set up distribution
  auto const [my_rank, num_ranks] = initialize_distribution();

  // kill off unused processes
  if (my_rank >= num_ranks)
  {
    finalize_distribution();
    return 0;
  }

  node_out() << "Branch: " << GIT_BRANCH << '\n';
  node_out() << "Commit Summary: " << GIT_COMMIT_HASH << GIT_COMMIT_SUMMARY
             << '\n';
  node_out() << "This executable was built on " << BUILD_TIME << '\n';

  // -- generate pde
  node_out() << "generating: pde..." << '\n';
  auto pde = make_PDE<prec>(cli_input);

  // do this only once to avoid confusion
  // if we ever do go to p-adaptivity (variable degree) we can change it then
  auto const degree = pde->get_dimensions()[0].get_degree();

  node_out() << "ASGarD problem configuration:" << '\n';
  node_out() << "  selected PDE: " << cli_input.get_pde_string() << '\n';
  node_out() << "  degree: " << degree << '\n';
  node_out() << "  N steps: " << opts.num_time_steps << '\n';
  node_out() << "  write freq: " << opts.wavelet_output_freq << '\n';
  node_out() << "  realspace freq: " << opts.realspace_output_freq << '\n';
  node_out() << "  implicit: " << opts.use_implicit_stepping << '\n';
  node_out() << "  full grid: " << opts.use_full_grid << '\n';
  node_out() << "  CFL number: " << cli_input.get_cfl() << '\n';
  node_out() << "  Poisson solve: " << opts.do_poisson_solve << '\n';
  node_out() << "  starting levels: ";
  node_out() << std::accumulate(
                    pde->get_dimensions().begin(), pde->get_dimensions().end(),
                    std::string(),
                    [](std::string const &accum, dimension<prec> const &dim) {
                      return accum + std::to_string(dim.get_level()) + " ";
                    })
             << '\n';
  node_out() << "  max adaptivity levels: " << opts.max_level << '\n';

  node_out() << "--- begin setup ---" << '\n';

  // -- create forward/reverse mapping between elements and indices,
  // -- along with a distribution plan. this is the adaptive grid.
  node_out() << "  generating: adaptive grid..." << '\n';

  adapt::distributed_grid adaptive_grid(*pde, opts);
  node_out() << "  degrees of freedom: "
             << adaptive_grid.size() *
                    static_cast<uint64_t>(std::pow(degree, pde->num_dims))
             << '\n';

  node_out() << "  generating: basis operator..." << '\n';
  auto const quiet = false;
  basis::wavelet_transform<prec, resource::host> const transformer(opts, *pde,
                                                                   quiet);
  // -- generate initial condition vector
  node_out() << "  generating: initial conditions..." << '\n';
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);
  node_out() << "  degrees of freedom (post initial adapt): "
             << adaptive_grid.size() *
                    static_cast<uint64_t>(std::pow(degree, pde->num_dims))
             << '\n';

  // -- generate and store coefficient matrices.
  node_out() << "  generating: coefficient matrices..." << '\n';
  generate_all_coefficients<prec>(*pde, transformer);

  // this is to bail out for further profiling/development on the setup routines
  if (opts.num_time_steps < 1)
    return 0;

  node_out() << "--- begin time loop staging ---" << '\n';

  // Our default device workspace size is 10GB - 12 GB DRAM on TitanV
  // - a couple GB for allocations not currently covered by the
  // workspace limit (including working batch).

  // This limit is only for the device workspace - the portion
  // of our allocation that will be resident on an accelerator
  // if the code is built for that.
  //
  // FIXME eventually going to be settable from the cmake
  static auto const default_workspace_MB = 10000;

  // FIXME currently used to check realspace transform only
  /* RAM on fusiont5 */
  static auto const default_workspace_cpu_MB = 187000;

  // -- setup output file and write initial condition
#ifdef ASGARD_IO_HIGHFIVE

  // initialize wavelet output
  auto output_dataset = initialize_output_file(initial_condition);

  // realspace solution vector - WARNING this is
  // currently infeasible to form for large problems
  auto const real_space_size = real_solution_size(*pde);
  fk::vector<prec> real_space(real_space_size);

  // temporary workspaces for the transform
  fk::vector<prec, mem_type::owner, resource::host> workspace(real_space_size *
                                                              2);
  std::array<fk::vector<prec, mem_type::view, resource::host>, 2>
      tmp_workspace = {
          fk::vector<prec, mem_type::view, resource::host>(workspace, 0,
                                                           real_space_size),
          fk::vector<prec, mem_type::view, resource::host>(
              workspace, real_space_size, real_space_size * 2 - 1)};
  // transform initial condition to realspace
  wavelet_to_realspace<prec>(*pde, initial_condition, adaptive_grid.get_table(),
                             transformer, default_workspace_cpu_MB,
                             tmp_workspace, real_space);

  // initialize realspace output
  auto const realspace_output_name = "asgard_realspace";
  auto output_dataset_real =
      initialize_output_file(real_space, "asgard_realspace");
#endif

  // -- time loop

  fk::vector<prec> f_val(initial_condition);
  node_out() << "--- begin time loop w/ dt " << pde->get_dt() << " ---\n";
  profiling::stop("Initialization");
  profiling::begin_iteration("Main");
  for (auto i = 0; i < opts.num_time_steps; ++i)
  {
    // take a time advance step
    auto const time          = (i + 1) * pde->get_dt();
    auto const update_system = i == 0;
    auto const method = opts.use_implicit_stepping ? time_advance::method::imp
                                                   : time_advance::method::exp;
    auto const time_str = opts.use_implicit_stepping ? "implicit_time_advance"
                                                     : "explicit_time_advance";
    auto const time_id = tools::timer.start(time_str);
    profiling::start(time_str);
    auto const sol = time_advance::adaptive_advance(
        method, *pde, adaptive_grid, transformer, opts, f_val, time,
        default_workspace_MB, update_system);
    f_val.resize(sol.size()) = sol;
    tools::timer.stop(time_id);
    profiling::stop(time_str);

    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln)
    {
      profiling::start("Compute_analytic_solnt");
      // get analytic solution at time(step+1)
      auto const subgrid           = adaptive_grid.get_subgrid(get_rank());
      auto const time_multiplier   = pde->exact_time(time + pde->get_dt());
      auto const analytic_solution = transform_and_combine_dimensions(
          *pde, pde->exact_vector_funcs, adaptive_grid.get_table(), transformer,
          subgrid.col_start, subgrid.col_stop, degree, time, time_multiplier);

      // calculate root mean squared error
      auto const diff = f_val - analytic_solution;
      auto const RMSE = [&diff]() {
        fk::vector<prec> squared(diff);
        std::transform(squared.begin(), squared.end(), squared.begin(),
                       [](prec const &elem) { return elem * elem; });
        auto const mean = std::accumulate(squared.begin(), squared.end(), 0.0) /
                          squared.size();
        return std::sqrt(mean);
      }();
      auto const relative_error = RMSE / inf_norm(analytic_solution) * 100;
      auto const [rmse_errors, relative_errors] =
          gather_errors(RMSE, relative_error);
      expect(rmse_errors.size() == relative_errors.size());
      for (int i = 0; i < rmse_errors.size(); ++i)
      {
        node_out() << "Errors for local rank: " << i << '\n';
        node_out() << "RMSE (numeric-analytic) [wavelet]: " << rmse_errors(i)
                   << '\n';
        node_out() << "Relative difference (numeric-analytic) [wavelet]: "
                   << relative_errors(i) << " %" << '\n';
      }
      profiling::stop("Compute_analytic_solnt");
    }
    else
    {
      node_out() << "No analytic solution found." << '\n';
    }

    // write output to file
#ifdef ASGARD_IO_HIGHFIVE
    if (opts.should_output_wavelet(i))
    {
      update_output_file(output_dataset, f_val);
    }

    /* transform from wavelet space to real space */
    if (opts.should_output_realspace(i))
    {
      wavelet_to_realspace<prec>(*pde, f_val, adaptive_grid.get_table(),
                                 transformer, default_workspace_cpu_MB,
                                 tmp_workspace, real_space);

      update_output_file(output_dataset_real, real_space,
                         realspace_output_name);
    }
#else
    ignore(default_workspace_cpu_MB);
#endif

    node_out() << "timestep: " << i << " complete" << '\n';
  }
  profiling::end_iteration("Main", default_workspace_MB, opts.num_time_steps);

  node_out() << "--- simulation complete ---" << '\n';

  auto const segment_size = element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  profiling::start("Gather_results");
  auto const final_result = gather_results(
      f_val, adaptive_grid.get_distrib_plan(), my_rank, segment_size);
  profiling::stop("Gather_results");

  node_out() << tools::timer.report() << '\n';

  tim::timemory_finalize();
  finalize_distribution();

  return 0;
}
