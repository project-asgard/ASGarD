#include "time_advance.hpp"
#include "distribution.hpp"
#include "element_table.hpp"
#include "fast_math.hpp"
#include "solver.hpp"

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in x.
template<typename P>
void explicit_time_advance(PDE<P> const &pde, element_table const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P> &host_space,
                           device_workspace<P> &dev_space,
                           std::vector<element_chunk> const &chunks,
                           distribution_plan const &plan, P const time,
                           P const dt)
{
  assert(time >= 0);
  assert(dt > 0);
  assert(static_cast<int>(unscaled_sources.size()) == pde.num_sources);

  fm::copy(host_space.x, host_space.x_orig);
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

  int const my_rank           = get_rank();
  element_subgrid const &grid = plan.at(my_rank);
  int const elem_size         = element_segment_size(pde);

  // FIXME remove prealloc and clearing
  // allocate batches
  // max number of elements in any chunk
  int const max_elems = num_elements_in_chunk(*std::max_element(
      chunks.begin(), chunks.end(),
      [](element_chunk const &a, element_chunk const &b) {
        return num_elements_in_chunk(a) < num_elements_in_chunk(b);
      }));
  // allocate batches for that size
  std::vector<batch_operands_set<P>> batches =
      allocate_batches<P>(pde, max_elems);

  apply_A(pde, table, grid, chunks, host_space, dev_space, batches);
  reduce_results(host_space.fx, host_space.reduced_fx, plan, my_rank);
  scale_sources(pde, unscaled_sources, host_space.scaled_source, time);
  fm::axpy(host_space.scaled_source, host_space.reduced_fx);
  exchange_results(host_space.reduced_fx, host_space.result_1, elem_size, plan,
                   my_rank);

  P const fx_scale_1 = a21 * dt;
  fm::axpy(host_space.result_1, host_space.x, fx_scale_1);

  apply_A(pde, table, grid, chunks, host_space, dev_space, batches);
  reduce_results(host_space.fx, host_space.reduced_fx, plan, my_rank);
  scale_sources(pde, unscaled_sources, host_space.scaled_source,
                time + c2 * dt);
  fm::axpy(host_space.scaled_source, host_space.reduced_fx);
  exchange_results(host_space.reduced_fx, host_space.result_2, elem_size, plan,
                   my_rank);

  fm::copy(host_space.x_orig, host_space.x);
  P const fx_scale_2a = a31 * dt;
  P const fx_scale_2b = a32 * dt;

  fm::axpy(host_space.result_1, host_space.x, fx_scale_2a);
  fm::axpy(host_space.result_2, host_space.x, fx_scale_2b);

  apply_A(pde, table, grid, chunks, host_space, dev_space, batches);
  reduce_results(host_space.fx, host_space.reduced_fx, plan, my_rank);
  scale_sources(pde, unscaled_sources, host_space.scaled_source,
                time + c3 * dt);
  fm::axpy(host_space.scaled_source, host_space.reduced_fx);

  exchange_results(host_space.reduced_fx, host_space.result_3, elem_size, plan,
                   my_rank);

  fm::copy(host_space.x_orig, host_space.x);
  P const scale_1 = dt * b1;
  P const scale_2 = dt * b2;
  P const scale_3 = dt * b3;

  fm::axpy(host_space.result_1, host_space.x, scale_1);
  fm::axpy(host_space.result_2, host_space.x, scale_2);
  fm::axpy(host_space.result_3, host_space.x, scale_3);
}

// scale source vectors for time
template<typename P>
static fk::vector<P> &
scale_sources(PDE<P> const &pde,
              std::vector<fk::vector<P>> const &unscaled_sources,
              fk::vector<P> &scaled_source, P const time)
{
  // zero out final vect
  fm::scal(static_cast<P>(0.0), scaled_source);
  // scale and accumulate all sources
  for (int i = 0; i < pde.num_sources; ++i)
  {
    fm::axpy(unscaled_sources[i], scaled_source,
             pde.sources[i].time_func(time));
  }
  return scaled_source;
}

// this function executes an implicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
void implicit_time_advance(PDE<P> const &pde, element_table const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P> &host_space,
                           std::vector<element_chunk> const &chunks,
                           P const time, P const dt, solve_opts const solver,
                           bool const update_system)
{
  assert(time >= 0);
  assert(dt > 0);
  assert(static_cast<int>(unscaled_sources.size()) == pde.num_sources);
  static fk::matrix<P, mem_type::owner, resource::host> A;
  static std::vector<int> ipiv;
  static bool first_time = true;

  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));
  int const A_size    = elem_size * table.size();

  scale_sources(pde, unscaled_sources, host_space.scaled_source, time + dt);
  host_space.x = host_space.x + host_space.scaled_source * dt;

  if (first_time || update_system)
  {
    A.clear_and_resize(A_size, A_size);
    for (auto const &chunk : chunks)
    {
      build_system_matrix(pde, table, chunk, A);
    }

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
      fm::gesv(A, host_space.x, ipiv);
      return;
      break;
    case solve_opts::gmres:
      ignore(ipiv);
      break;
    }
    first_time = false;
  } // end first time/update system

  switch (solver)
  {
  case solve_opts::direct:
    fm::getrs(A, host_space.x, ipiv);
    break;
  case solve_opts::gmres:
    P const tolerance  = std::is_same_v<float, P> ? 1e-6 : 1e-12;
    int const restart  = A.ncols();
    int const max_iter = A.ncols();
    fm::scal(static_cast<P>(0.0), host_space.result_1);
    solver::simple_gmres(A, host_space.result_1, host_space.x, fk::matrix<P>(),
                         restart, max_iter, tolerance);
    fm::copy(host_space.result_1, host_space.x);
    break;
  }
}

template void
explicit_time_advance(PDE<double> const &pde, element_table const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double> &host_space,
                      device_workspace<double> &dev_space,
                      std::vector<element_chunk> const &chunks,
                      distribution_plan const &plan, double const time,
                      double const dt);

template void
explicit_time_advance(PDE<float> const &pde, element_table const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float> &host_space,
                      device_workspace<float> &dev_space,
                      std::vector<element_chunk> const &chunks,
                      distribution_plan const &plan, float const time,
                      float const dt);

template void
implicit_time_advance(PDE<double> const &pde, element_table const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double> &host_space,
                      std::vector<element_chunk> const &chunks,
                      double const time, double const dt,
                      solve_opts const solver, bool const update_system);

template void
implicit_time_advance(PDE<float> const &pde, element_table const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float> &host_space,
                      std::vector<element_chunk> const &chunks,
                      float const time, float const dt, solve_opts const solver,
                      bool const update_system);
