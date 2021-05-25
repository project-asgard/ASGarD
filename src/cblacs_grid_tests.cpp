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

TEST_CASE("", "[cblacs_grid]")
{
  int myrank    = get_rank();
  int num_ranks = get_num_ranks();
  cblacs_grid grid;
  int myrow      = grid.get_myrow();
  int mycol      = grid.get_mycol();
  int local_rows = grid.local_rows(3, 256);
  int local_cols = grid.local_cols(3, 256);
  // std::cout << context << ' ' << myrow << ' ' << mycol << ' ' << local_rows
  // << ' ' << local_cols << '\n'; CHECK(false);
}
