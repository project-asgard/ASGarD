#include <cassert>
#include <cmath>

extern "C"
{
  int numroc_(int *, int *, int *, int *, int *);
  void Cblacs_get(int, int, int *);
  void Cblacs_gridinit(int *, const char *, int, int);
  void Cblacs_gridinfo(int, int *, int *, int *, int *);
  void Cblacs_gridexit(int);
  void Cblacs_pinfo(int *, int *);
  void Cblacs_exit(int);
}

class cblacs_grid
{
public:
  cblacs_grid()
  {
    int i_negone{-1}, i_zero{0};
    int myid, numproc;
    Cblacs_pinfo(&myid, &numproc);
    for (npcol_ = std::sqrt(numproc) + 1; npcol_ >= 1; npcol_--)
    {
      nprow_        = numproc / npcol_;
      bool is_found = ((nprow_ * npcol_) == numproc);
      if (is_found)
        break;
    };
    assert((nprow_ >= 1) && (npcol_ >= 1) && (nprow_ * npcol_ == numproc));
    Cblacs_get(i_negone, i_zero, &ictxt_);
    Cblacs_gridinit(&ictxt_, "R", nprow_, npcol_);
    Cblacs_gridinfo(ictxt_, &nprow_, &npcol_, &myrow_, &mycol_);
  }
  int get_context() const { return ictxt_; }
  int get_myrow() const { return myrow_; }
  int get_mycol() const { return mycol_; }
  int local_rows(int m, int mb)
  {
    int i_zero{0};
    return numroc_(&m, &mb, &myrow_, &i_zero, &nprow_);
  }
  int local_cols(int n, int nb)
  {
    int i_zero{0};
    return numroc_(&n, &nb, &mycol_, &i_zero, &npcol_);
  }
  ~cblacs_grid() { Cblacs_gridexit(ictxt_); }

private:
  int ictxt_, nprow_{1}, npcol_, myrow_, mycol_;
};
