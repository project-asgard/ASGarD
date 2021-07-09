#pragma once

class cblacs_grid
{
public:
  cblacs_grid();
  int get_context() const { return ictxt_; }
  int get_myrow() const { return myrow_; }
  int get_mycol() const { return mycol_; }
  int local_rows(int m, int mb, bool distributed = true);
  int local_cols(int n, int nb);
  ~cblacs_grid();

private:
  int ictxt_, nprow_{1}, npcol_, myrow_, mycol_;
};
