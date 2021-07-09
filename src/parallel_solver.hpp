#pragma once
#include "cblacs_grid.hpp"
#include "tensors.hpp"

enum DESC_VARS : int
{
  DTYPE_ = 0,
  CTXT_  = 1,
  M_     = 2,
  N_     = 3,
  MB_    = 4,
  NB_    = 5,
  RSRC_  = 6,
  CSRC_  = 7,
  LLD_   = 8,
  DLEN_  = 9
};

template<typename P>
class parallel_solver
{
public:
  parallel_solver(int mb = 256, int nb = 256) : mb_{mb}, nb_{nb} {}

  void
  gather_matrix(P *A, int *descA, P *A_distr, int *descA_distr, int n, int m);
  void
  scatter_matrix(P *A, int *descA, P *A_distr, int *descA_distr, int n, int m);

  void descinit(int *descA, int n, int m);
  void descinit_distr(int *descA_distr, int n, int m);

  void resize(fk::matrix<P> &A_distr, int n, int m);
  void resize(fk::vector<P> &A_distr, int m);

private:
  cblacs_grid grid_;
  int mb_, nb_;
};
