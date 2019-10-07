#pragma once
#include "batch.hpp"
#include "chunk.hpp"
#include "program_options.hpp"
#include "tensors.hpp"

// this function executes a time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P, typename T>
void explicit_time_advance(PDE<P> const &pde, element_table<T> const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P, T> &host_space,
                           rank_workspace<P> &rank_space,
                           std::vector<element_chunk> chunks, P const time,
                           P const dt);

template<typename P, typename T>
void implicit_time_advance(PDE<P> const &pde, element_table<T> const &table,
                           std::vector<fk::vector<P>> const &unscaled_sources,
                           host_workspace<P,T> &host_space,
                           std::vector<element_chunk> const &chunks,
                           P const time, P const dt, bool update_system = true);
extern template void
explicit_time_advance(PDE<float> const &pde, element_table<int> const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float, int> &host_space,
                      rank_workspace<float> &rank_space,
                      std::vector<element_chunk> chunks, float const time,
                      float const dt);
extern template void
explicit_time_advance(PDE<float> const &pde, element_table<int> const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float, int> &host_space,
                      rank_workspace<float> &rank_space,
                      std::vector<element_chunk> chunks, float const time,
                      float const dt);
extern template void
explicit_time_advance(PDE<float> const &pde,
                      element_table<int64_t> const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float, int64_t> &host_space,
                      rank_workspace<float> &rank_space,
                      std::vector<element_chunk> chunks, float const time,
                      float const dt);

extern template void
explicit_time_advance(PDE<double> const &pde, element_table<int> const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double, int> &host_space,
                      rank_workspace<double> &rank_space,
                      std::vector<element_chunk> chunks, double const time,
                      double const dt);
extern template void
explicit_time_advance(PDE<double> const &pde,
                      element_table<int64_t> const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double, int64_t> &host_space,
                      rank_workspace<double> &rank_space,
                      std::vector<element_chunk> chunks, double const time,
                      double const dt);

extern template void
implicit_time_advance(PDE<double> const &pde, element_table<int> const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double,int> &host_space,
                      std::vector<element_chunk> const &chunks,
                      double const time, double const dt,
                      bool update_system = true);
extern template void
implicit_time_advance(PDE<double> const &pde, element_table<int64_t> const &table,
                      std::vector<fk::vector<double>> const &unscaled_sources,
                      host_workspace<double,int64_t> &host_space,
                      std::vector<element_chunk> const &chunks,
                      double const time, double const dt,
                      bool update_system = true);

extern template void
implicit_time_advance(PDE<float> const &pde, element_table<int> const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float,int> &host_space,
                      std::vector<element_chunk> const &chunks,
                      float const time, float const dt,
                      bool update_system = true);
extern template void
implicit_time_advance(PDE<float> const &pde, element_table<int64_t> const &table,
                      std::vector<fk::vector<float>> const &unscaled_sources,
                      host_workspace<float,int64_t> &host_space,
                      std::vector<element_chunk> const &chunks,
                      float const time, float const dt,
                      bool update_system = true);
