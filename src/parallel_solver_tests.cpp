#include "distribution.hpp"
#include "parallel_solver.hpp"
#include "tests_general.hpp"

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEMPLATE_TEST_CASE("", "[parallel_solver]", float, double)
{
  int myrank    = get_rank();
  int num_ranks = get_num_ranks();

  fk::matrix<TestType> A{
      {0., 0., 1., 1.}, {0., 0., 1., 1.}, {2., 2., 3., 3.}, {2., 2., 3., 3.}};

  if (myrank != 0)
    A.clear_and_resize(0,0);

  fk::vector<TestType> const B{0., 0., 2., 2.};

  parallel_solver<TestType> solver(2, 2);

  int n = 4;
  int m = 4;
  REQUIRE(n == m);
  fk::matrix<TestType> A_distr;
  solver.resize(A_distr, n, m);
  if (num_ranks == 1)
  {
    REQUIRE(A_distr.size() == A.size());
  }
  else
  {
    REQUIRE(A_distr.size() == 4);
  }

  int descA[9], descA_distr[9];
  solver.descinit(descA, n, n);
  solver.descinit_distr(descA_distr, n, n);
  solver.scatter_matrix(A.data(), descA, A_distr.data(), descA_distr, n, n);

  if (num_ranks == 1)
  {
    for (int i = 0; i < m; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
        REQUIRE_THAT(A(i, j), Catch::Matchers::WithinRel(A_distr(i, j),
                                                         TestType{0.001}));
      }
    }
  }
  else
  {
    for (int i = 0; i < 2; ++i)
    {
      for (int j = 0; j < 2; ++j)
      {
        REQUIRE_THAT(A_distr(i, j),
                     Catch::Matchers::WithinRel(myrank, TestType{0.001}));
      }
    }
  }

  n = B.size();
  fk::vector<TestType> B_distr;
  solver.resize(B_distr, n);
  if (num_ranks == 1)
  {
    REQUIRE(B_distr.size() == B.size());
  }
  else
  {
    REQUIRE(B_distr.size() == 2);
  }

  int descB[9], descB_distr[9];
  solver.descinit(descB, 1, n);
  solver.descinit_distr(descB_distr, 1, n);
  solver.scatter_matrix(B.data(), descB, B_distr.data(), descB_distr, 1, n);

  if (num_ranks == 1)
  {
    for (int i = 0; i < n; ++i)
    {
      REQUIRE_THAT(B(i),
                   Catch::Matchers::WithinRel(B_distr(i), TestType{0.001}));
    }
  }
  else if (myrank % 2 == 0)
  {
    for (int i = 0; i < 2; ++i)
    {
      REQUIRE_THAT(B_distr(i),
                   Catch::Matchers::WithinRel(myrank, TestType{0.001}));
    }
  }
}
