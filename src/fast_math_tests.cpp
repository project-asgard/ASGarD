#include "fast_math.hpp"
#include "distribution.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"
#include <cmath>
#include <numeric>

struct distribution_test_init
{
  distribution_test_init() { initialize_distribution(); }
  ~distribution_test_init() { finalize_distribution(); }
};

#ifdef ASGARD_USE_MPI
static distribution_test_init const distrib_test_info;
#endif

TEMPLATE_TEST_CASE("LU Routines", "[fast_math]", double, float)
{
  fk::matrix<TestType> const A_gold{
      {3.383861628748717e+00, 1.113343240310116e-02, 2.920740795411032e+00},
      {3.210305545769361e+00, 3.412141162288144e+00, 3.934494120167269e+00},
      {1.723479266939425e+00, 1.710451084172946e+00, 4.450671104482062e+00}};

  fk::matrix<TestType> const L_gold{
      {1.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00},
      {9.487106442223131e-01, 1.000000000000000e+00, 0.000000000000000e+00},
      {5.093232099968379e-01, 5.011733347067855e-01, 1.000000000000000e+00}};

  fk::matrix<TestType> const U_gold{
      {3.383861628748717e+00, 1.113343240310116e-02, 2.920740795411032e+00},
      {0.000000000000000e+00, 3.401578756460592e+00, 1.163556238546477e+00},
      {0.000000000000000e+00, 0.000000000000000e+00, 2.379926666803375e+00}};

  fk::matrix<TestType> const I_gold{{1.00000, 0.00000, 0.00000},
                                    {0.00000, 1.00000, 0.00000},
                                    {0.00000, 0.00000, 1.00000}};

  fk::vector<TestType> const B_gold{
      2.084406360034887e-01, 6.444769305362776e-01, 3.687335330031937e-01};
  fk::vector<TestType> const X_gold{
      4.715561567725287e-02, 1.257695999382253e-01, 1.625351700791827e-02};

  fk::vector<TestType> const B1_gold{
      9.789303188021963e-01, 8.085725142873675e-01, 7.370498473207234e-01};
  fk::vector<TestType> const X1_gold{
      1.812300946484165e-01, -7.824949213916167e-02, 1.254969087137521e-01};

  fk::matrix<TestType> const LU_gold = L_gold + U_gold - I_gold;

#ifdef ASGARD_USE_SLATE

  SECTION("slate_gesv and slate_getrs")
  {
    fk::matrix<TestType> const A_copy = A_gold;
    std::vector<int> ipiv(A_copy.nrows());
    fk::vector<TestType> x = B_gold;

    fm::gesv(A_copy, x, ipiv, solve_opts::slate);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-16 : 1e-7;
    int rank = get_rank();
    if(rank == 0) {
        rmse_comparison(A_copy, LU_gold, tol_factor);
        rmse_comparison(x, X_gold, tol_factor);
        x = B1_gold;
        fm::getrs(A_copy, x, ipiv, solve_opts::slate);
        rmse_comparison(x, X1_gold, tol_factor);
    }
  }
#endif
}
