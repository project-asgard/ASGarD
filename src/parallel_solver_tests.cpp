#include "parallel_solver.hpp"
#include "tests_general.hpp"

TEMPLATE_TEST_CASE("", "[parallel_solver]", float, double)
{
  parallel_solver<TestType> solver;
}
