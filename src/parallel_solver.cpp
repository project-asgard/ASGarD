#include "parallel_solver.hpp"

extern "C"
{
  void descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *irsrc,
                 int *icsrc, int *ictxt, int *lld, int *info);
  void pdgeadd_(char *, int *, int *, double *, double *, int *, int *, int *,
                double *, double *, int *, int *, int *);
  void psgeadd_(char *, int *, int *, float *, float *, int *, int *, int *,
                float *, float *, int *, int *, int *);
}

template<typename P>
void parallel_solver<P>::gather_matrix(P *A, int *descA, P *A_distr,
                                       int *descA_distr, int n, int m)
{
  // Useful constants
  P zero{0.0E+0}, one{1.0E+0};
  int i_one{1};
  char N{'N'};
  // Call pdgeadd_ to distribute matrix (i.e. copy A into A_distr)
  if constexpr (std::is_same<P, double>::value)
  {
    pdgeadd_(&N, &m, &n, &one, A_distr, &i_one, &i_one, descA_distr, &zero, A,
             &i_one, &i_one, descA);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    psgeadd_(&N, &m, &n, &one, A_distr, &i_one, &i_one, descA_distr, &zero, A,
             &i_one, &i_one, descA);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "geadd not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void parallel_solver<P>::scatter_matrix(P *A, int *descA, P *A_distr,
                                        int *descA_distr, int n, int m)
{
  // Useful constants
  P zero{0.0E+0}, one{1.0E+0};
  int i_one{1};
  char N{'N'};
  // Call pdgeadd_ to distribute matrix (i.e. copy A into A_distr)
  if constexpr (std::is_same<P, double>::value)
  {
    pdgeadd_(&N, &m, &n, &one, A, &i_one, &i_one, descA, &zero, A_distr, &i_one,
             &i_one, descA_distr);
  }
  else if constexpr (std::is_same<P, float>::value)
  {
    psgeadd_(&N, &m, &n, &one, A, &i_one, &i_one, descA, &zero, A_distr, &i_one,
             &i_one, descA_distr);
  }
  else
  { // not instantiated; should never be reached
    std::cerr << "geadd not implemented for non-floating types" << '\n';
    expect(false);
  }
}

template<typename P>
void parallel_solver<P>::descinit(int *descA, int n, int m)
{
  int i_zero{0}, info;
  int ictxt = grid_.get_context();
  int lld   = std::max(1, grid_.local_rows(m, n, false));
  descinit_(descA, &m, &n, &m, &n, &i_zero, &i_zero, &ictxt, &lld, &info);
}

template<typename P>
void parallel_solver<P>::descinit_distr(int *descA_distr, int n, int m)
{
  int i_zero{0}, info;
  int ictxt = grid_.get_context();
  int lld   = std::max(1, grid_.local_rows(m, mb_));
  descinit_(descA_distr, &m, &n, &mb_, &nb_, &i_zero, &i_zero, &ictxt, &lld,
            &info);
}

template<typename P>
void parallel_solver<P>::resize(fk::matrix<P> &A_distr, int n, int m)
{
  int mp = grid_.local_rows(m, mb_);
  int nq = grid_.local_cols(n, nb_);
  A_distr.clear_and_resize(mp, nq);
}

template<typename P>
void parallel_solver<P>::resize(fk::vector<P> &A_distr, int m)
{
  int mp = grid_.local_rows(m, mb_);
  A_distr.resize(mp);
}

template class parallel_solver<float>;
template class parallel_solver<double>;
