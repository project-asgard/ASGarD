#include "time_advance.hpp"
#include "adapt.hpp"
#include "batch.hpp"
#include "boundary_conditions.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "fast_math.hpp"
#include "solver.hpp"
#include "tools.hpp"
#include <limits.h>

namespace time_advance
{
template<typename P>
static fk::vector<P>
get_sources(PDE<P> const &pde, adapt::distributed_grid<P> const &grid,
            basis::wavelet_transform<P, resource::host> const &transformer,
            P const time)
{
  auto const my_subgrid = grid.get_subgrid(get_rank());
  // FIXME assume uniform degree
  auto const degree = pde.get_dimensions()[0].get_degree();
  auto const dof    = std::pow(degree, pde.num_dims) * my_subgrid.nrows();
  fk::vector<P> sources(dof);
  for (auto const &source : pde.sources)
  {
    auto const source_vect = transform_and_combine_dimensions(
        pde, source.source_funcs, grid.get_table(), transformer,
        my_subgrid.row_start, my_subgrid.row_stop, degree, time,
        source.time_func(time));
    fm::axpy(source_vect, sources);
  }
  return sources;
}

// FIXME want to change how sources/bcs are handled
template<typename P>
fk::vector<P>
adaptive_advance(method const step_method, PDE<P> &pde,
                 adapt::distributed_grid<P> &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts, fk::vector<P> const &x_orig,
                 P const time, int const workspace_size_MB,
                 bool const update_system)
{
  if (!program_opts.do_adapt_levels)
  {
    auto const my_subgrid     = adaptive_grid.get_subgrid(get_rank());
    auto const unscaled_parts = boundary_conditions::make_unscaled_bc_parts(
        pde, adaptive_grid.get_table(), transformer, my_subgrid.row_start,
        my_subgrid.row_stop);
    return (step_method == method::exp)
               ? explicit_advance(pde, adaptive_grid, transformer, program_opts,
                                  unscaled_parts, x_orig, workspace_size_MB,
                                  time)
               : implicit_advance(pde, adaptive_grid, transformer,
                                  unscaled_parts, x_orig, time,
                                  program_opts.solver, update_system);
  }

  // coarsen
  auto const old_size = adaptive_grid.size();
  auto y = adaptive_grid.coarsen_solution(pde, x_orig, program_opts);
  node_out() << " adapt -- coarsened grid from " << old_size << " -> "
             << adaptive_grid.size() << " elems\n";

  // refine
  auto refining = true;
  while (refining)
  {
    // update boundary conditions
    auto const my_subgrid     = adaptive_grid.get_subgrid(get_rank());
    auto const unscaled_parts = boundary_conditions::make_unscaled_bc_parts(
        pde, adaptive_grid.get_table(), transformer, my_subgrid.row_start,
        my_subgrid.row_stop);

    // take a probing refinement step
    auto const y_stepped =
        (step_method == method::exp)
            ? explicit_advance(pde, adaptive_grid, transformer, program_opts,
                               unscaled_parts, y, workspace_size_MB, time)
            : implicit_advance(pde, adaptive_grid, transformer, unscaled_parts,
                               y, time, program_opts.solver, update_system);

    auto const old_plan = adaptive_grid.get_distrib_plan();
    auto const old_size = adaptive_grid.size();
    auto const y_refined =
        adaptive_grid.refine_solution(pde, y_stepped, program_opts);
    refining = static_cast<bool>(
        get_global_max(static_cast<float>(y_stepped.size() != y_refined.size()),
                       adaptive_grid.get_distrib_plan()));

    node_out() << " adapt -- refined grid from " << old_size << " -> "
               << adaptive_grid.size() << " elems\n";

    if (!refining)
    {
      y.resize(y_stepped.size()) = y_stepped;
    }
    else
    {
      auto const y1 =
          adaptive_grid.redistribute_solution(y, old_plan, old_size);
      y.resize(y1.size()) = y1;
    }
  }

  return y;
}

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in x.
template<typename P>
fk::vector<P>
explicit_advance(PDE<P> const &pde,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 options const &program_opts,
                 std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                 fk::vector<P> const &x_orig, int const workspace_size_MB,
                 P const time)
{
  auto const &table    = adaptive_grid.get_table();
  auto const &plan     = adaptive_grid.get_distrib_plan();
  auto const &grid     = adaptive_grid.get_subgrid(get_rank());
  auto const elem_size = element_segment_size(pde);
  auto const dt        = pde.get_dt();
  auto const col_size  = elem_size * static_cast<int64_t>(grid.ncols());
  expect(x_orig.size() == col_size);
  auto const row_size = elem_size * static_cast<int64_t>(grid.nrows());
  expect(col_size < INT_MAX);
  expect(row_size < INT_MAX);

  // time advance working vectors
  // input vector for apply_A
  fk::vector<P> x(x_orig);
  // a buffer for reducing across subgrid row
  fk::vector<P> reduced_fx(row_size);

  expect(time >= 0);
  expect(dt > 0);

  // see
  // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
  P const a21 = 0.5;
  P const a31 = -1.0;
  P const a32 = 2.0;
  P const b1  = 1.0 / 6.0;
  P const b2  = 2.0 / 3.0;
  P const b3  = 1.0 / 6.0;
  P const c2  = 1.0 / 2.0;
  P const c3  = 1.0;

  // FIXME eventually want to extract RK step into function
  // -- RK step 1
  auto const apply_id = tools::timer.start("kronmult_setup");
  auto fx =
      kronmult::execute(pde, table, program_opts, grid, workspace_size_MB, x);

  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources > 0)
  {
    auto const sources = get_sources(pde, adaptive_grid, transformer, time);
    fm::axpy(sources, reduced_fx);
  }

  auto const bc0 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time);
  fm::axpy(bc0, reduced_fx);

  // FIXME I eventually want to return a vect here
  fk::vector<P> rk_1(x_orig.size());
  exchange_results(reduced_fx, rk_1, elem_size, plan, get_rank());
  P const rk_scale_1 = a21 * dt;
  fm::axpy(rk_1, x, rk_scale_1);

  // -- RK step 2
  tools::timer.start(apply_id);
  fx = kronmult::execute(pde, table, program_opts, grid, workspace_size_MB, x);
  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources > 0)
  {
    auto const sources =
        get_sources(pde, adaptive_grid, transformer, time + c2 * dt);
    fm::axpy(sources, reduced_fx);
  }

  fk::vector<P> const bc1 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + c2 * dt);
  fm::axpy(bc1, reduced_fx);

  fk::vector<P> rk_2(x_orig.size());
  exchange_results(reduced_fx, rk_2, elem_size, plan, get_rank());

  fm::copy(x_orig, x);
  P const rk_scale_2a = a31 * dt;
  P const rk_scale_2b = a32 * dt;

  fm::axpy(rk_1, x, rk_scale_2a);
  fm::axpy(rk_2, x, rk_scale_2b);

  // -- RK step 3
  tools::timer.start(apply_id);
  fx = kronmult::execute(pde, table, program_opts, grid, workspace_size_MB, x);
  tools::timer.stop(apply_id);
  reduce_results(fx, reduced_fx, plan, get_rank());

  if (pde.num_sources > 0)
  {
    auto const sources =
        get_sources(pde, adaptive_grid, transformer, time + c3 * dt);
    fm::axpy(sources, reduced_fx);
  }

  auto const bc2 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + c3 * dt);
  fm::axpy(bc2, reduced_fx);

  fk::vector<P> rk_3(x_orig.size());
  exchange_results(reduced_fx, rk_3, elem_size, plan, get_rank());

  // -- finish
  fm::copy(x_orig, x);
  P const scale_1 = dt * b1;
  P const scale_2 = dt * b2;
  P const scale_3 = dt * b3;

  fm::axpy(rk_1, x, scale_1);
  fm::axpy(rk_2, x, scale_2);
  fm::axpy(rk_3, x, scale_3);

  return x;
}

// this function executes an implicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
fk::vector<P>
implicit_advance(PDE<P> const &pde,
                 adapt::distributed_grid<P> const &adaptive_grid,
                 basis::wavelet_transform<P, resource::host> const &transformer,
                 std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                 fk::vector<P> const &x_orig, P const time,
                 solve_opts const solver, bool const update_system)
{
  expect(time >= 0);

  static fk::matrix<P, mem_type::owner, resource::host> A;
  static std::vector<int> ipiv;
  static bool first_time = true;

  auto const &table   = adaptive_grid.get_table();
  auto const dt       = pde.get_dt();
  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));
  int const A_size    = elem_size * table.size();

  fk::vector<P> x(x_orig);

  if (pde.num_sources > 0)
  {
    auto const sources =
        get_sources(pde, adaptive_grid, transformer, time + dt);
    fm::axpy(sources, x, dt);
  }

  /* add the boundary condition */
  auto const &grid = adaptive_grid.get_subgrid(get_rank());

  auto const bc = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + dt);
  fm::axpy(bc, x, dt);

  if (first_time || update_system)
  {
    first_time = false;

    A.clear_and_resize(A_size, A_size);
    build_system_matrix(pde, table, A);

    // AA = I - dt*A;
    for (int i = 0; i < A.nrows(); ++i)
    {
      for (int j = 0; j < A.ncols(); ++j)
      {
        A(i, j) *= -dt;
      }
      A(i, i) += 1.0;
    }

    switch (solver)
    {
    case solve_opts::direct:
      if (ipiv.size() != static_cast<unsigned long>(A.nrows()))
        ipiv.resize(A.nrows());
      fm::gesv(A, x, ipiv, solver);
      return x;
      break;
    case solve_opts::slate:
      if (ipiv.size() != static_cast<unsigned long>(A.nrows()))
        ipiv.resize(A.nrows());
      fm::gesv(A, x, ipiv, solver);
      return x;
      break;
    case solve_opts::gmres:
      ignore(ipiv);
      break;
    }
  } // end first time/update system

  switch (solver)
  {
  case solve_opts::direct:
    fm::getrs(A, x, ipiv, solver);
    return x;
    break;
  case solve_opts::slate:
    fm::getrs(A, x, ipiv, solver);
    return x;
    break;
  case solve_opts::gmres:
    P const tolerance  = std::is_same_v<float, P> ? 1e-6 : 1e-12;
    int const restart  = A.ncols();
    int const max_iter = A.ncols();
    fk::vector<P> fx(x.size());
    solver::simple_gmres(A, fx, x, fk::matrix<P>(), restart, max_iter,
                         tolerance);
    return fx;
    break;
  }
  return x;
}

#define X(T)                                                             \
  template fk::vector<T> adaptive_advance(                               \
      method const step_method, PDE<T> &pde,                             \
      adapt::distributed_grid<T> &adaptive_grid,                         \
      basis::wavelet_transform<T, resource::host> const &transformer,    \
      options const &program_opts, fk::vector<T> const &x, T const time, \
      int const workspace_size_MB, bool const update_system);
#include "type_list_float.inc"
#undef X

#define X(T)                                                              \
  template fk::vector<T> explicit_advance(                                \
      PDE<T> const &pde, adapt::distributed_grid<T> const &adaptive_grid, \
      basis::wavelet_transform<T, resource::host> const &transformer,     \
      options const &program_opts,                                        \
      std::array<unscaled_bc_parts<T>, 2> const &unscaled_parts,          \
      fk::vector<T> const &x, int const workspace_size_MB, T const time);
#include "type_list_float.inc"
#undef X

#define X(T)                                                                  \
  template fk::vector<T> implicit_advance(                                    \
      PDE<T> const &pde, adapt::distributed_grid<T> const &adaptive_grid,     \
      basis::wavelet_transform<T, resource::host> const &transformer,         \
      std::array<unscaled_bc_parts<T>, 2> const &unscaled_parts,              \
      fk::vector<T> const &host_space, T const time, solve_opts const solver, \
      bool const update_system);
#include "type_list_float.inc"
#undef X

} // namespace time_advance
