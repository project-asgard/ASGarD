#pragma once

#include "batch.hpp"
#include "chunk.hpp"
#include "distribution.hpp"
#include "pde_base.hpp"
#include "tensors.hpp"

// perform GMRES using apply_A for tensor encoded matrix*vector
// returns the residual
// approximate solution stored in host_space.x
template<typename P>
P gmres(PDE<P> const &pde, element_table const &elem_table,
        distribution_plan const &plan, std::vector<element_chunk> const &chunks,
        host_workspace<P> &host_space, rank_workspace<P> &rank_space,
        std::vector<batch_operands_set<P>> &batches, P const dt);

// simple, node-local test version
template<typename P>
P simple_gmres(fk::matrix<P> const &A, fk::vector<P> const &x, fk::vector<P> &b,
               fk::matrix<P> const &M, int const restart, int const max_iter,
               P const tolerance);

extern template float
gmres(PDE<float> const &pde, element_table const &elem_table,
      distribution_plan const &plan, std::vector<element_chunk> const &chunks,
      host_workspace<float> &host_space, rank_workspace<float> &rank_space,
      std::vector<batch_operands_set<float>> &batches, float const dt);

extern template double
gmres(PDE<double> const &pde, element_table const &elem_table,
      std::vector<element_chunk> const &chunks, distribution_plan const &plan,
      host_workspace<double> &host_space, rank_workspace<double> &rank_space,
      std::vector<batch_operands_set<double>> &batches, double const dt);

extern template float float
simple_gmres(fk::matrix<float> const &A, fk::vector<float> const &x,
             fk::vector<float> &b, fk::matrix<float> const &M,
             int const restart, int const max_iter, float const tolerance);

extern template double double
simple_gmres(fk::matrix<double> const &A, fk::vector<double> const &x,
             fk::vector<double> &b, fk::matrix<double> const &M,
             int const restart, int const max_iter, double const tolerance);