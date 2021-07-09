#include "cblacs_grid.hpp"
#include "distribution.hpp"
#include "tests_general.hpp"

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEST_CASE("Generating a cblacs grid.", "[cblacs_grid]")
{
  int myrank    = get_rank();
  int num_ranks = get_num_ranks();
  int npcol     = std::sqrt(num_ranks);
  cblacs_grid grid;
  int myrow = grid.get_myrow();
  int mycol = grid.get_mycol();
  REQUIRE(myrank / npcol == myrow);
  REQUIRE(myrank % npcol == mycol);

  int local_rows = grid.local_rows(4, 1);
  int local_cols = grid.local_cols(4, 1);
  if (num_ranks == 4)
  {
    // 4 elements on each process
    REQUIRE(local_rows * local_cols == 4);
  }
  else
  {
    // 16 elements on one process
    REQUIRE(local_rows * local_cols == 16);
  }

  local_rows = grid.local_rows(4, 256);
  local_cols = grid.local_cols(4, 256);
  REQUIRE(myrank / npcol == myrow);
  REQUIRE(myrank % npcol == mycol);
  if (myrank == 0)
  {
    // 16 elements on zeroth proces
    REQUIRE(local_rows * local_cols == 16);
  }
  else
  {
    // None on other processes
    REQUIRE(local_rows * local_cols == 0);
  }
}
