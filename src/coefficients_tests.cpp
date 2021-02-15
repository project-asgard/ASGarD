#include "coefficients.hpp"
#include "matlab_utilities.hpp"
#include "pde.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"

template<typename P>
void test_coefficients(parser const &parse, std::string const &gold_path,
                       P const tol_factor = 1e-15, bool const rotate = true)
{
  auto pde = make_PDE<P>(parse);
  options const opts(parse);
  basis::wavelet_transform<P, resource::host> const transformer(opts, *pde);
  P const time = 1.0;
  generate_all_coefficients(*pde, transformer, time, rotate);

  auto const lev_string = std::accumulate(
      pde->get_dimensions().begin(), pde->get_dimensions().end(), std::string(),
      [](std::string const &accum, dimension<P> const &dim) {
        return accum + std::to_string(dim.get_level()) + "_";
      });

  // FIXME assume uniform degree across dimensions here
  auto const filename_base = gold_path + "_l" + lev_string + "d" +
                             std::to_string(parse.get_degree()) + "_";

  for (auto d = 0; d < pde->num_dims; ++d)
  {
    for (auto t = 0; t < pde->num_terms; ++t)
    {
      auto const filename = filename_base + std::to_string(t + 1) + "_" +
                            std::to_string(d + 1) + ".dat";
      auto const gold = fk::matrix<P>(read_matrix_from_txt_file(filename));
      auto const full_coeff = pde->get_coefficients(t, d).clone_onto_host();

      auto const &dim = pde->get_dimensions()[d];
      auto const degrees_freedom_1d =
          dim.get_degree() * fm::two_raised_to(dim.get_level());
      fk::matrix<P, mem_type::const_view> const test(
          full_coeff, 0, degrees_freedom_1d - 1, 0, degrees_freedom_1d - 1);
      rmse_comparison(gold, test, tol_factor);
    }
  }
}

TEMPLATE_TEST_CASE("diffusion 2 (single term)", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::diffusion_2;
  auto const gold_path =
      "../testing/generated-inputs/coefficients/diffusion2_coefficients";
  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-13 : 1e-4;

  SECTION("level 3, degree 5")
  {
    auto const degree = 5;
    auto const levels = fk::vector<int>{3, 3};
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform level: levels 2, 3, degree 5")
  {
    auto const degree = 5;
    auto const levels = fk::vector<int>{2, 3};
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("diffusion 1 (single term)", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::diffusion_1;
  auto const gold_path =
      "../testing/generated-inputs/coefficients/diffusion1_coefficients";
  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-12 : 1e-3;

  SECTION("level 5, degree 6")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 6;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 1 (single term)", "[coefficients]", double,
                   float)
{
  auto const pde_choice = PDE_opts::continuity_1;
  auto const gold_path =
      "../testing/generated-inputs/coefficients/continuity1_coefficients";
  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-4;

  SECTION("level 2, degree 2 (default)")
  {
    auto const levels = fk::vector<int>{2};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 2 terms", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::continuity_2;
  auto const gold_path =
      "../testing/generated-inputs/coefficients/continuity2_coefficients";
  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-6;

  SECTION("level 4, degree 3")
  {
    auto const levels = fk::vector<int>{4, 4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform level: levels 4, 5, degree 3")
  {
    auto const levels = fk::vector<int>{4, 5};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 3 terms", "[coefficients]", double, float)
{
  auto const gold_path =
      "../testing/generated-inputs/coefficients/continuity3_coefficients";
  auto const pde_choice     = PDE_opts::continuity_3;
  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-5;

  SECTION("level 4, degree 4")
  {
    auto const levels = fk::vector<int>{4, 4, 4};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non uniform level: levels 2, 3, 2, degree 4")
  {
    auto const levels = fk::vector<int>{2, 3, 2};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("continuity 6 terms", "[coefficients]", double, float)
{
  auto const gold_path =
      "../testing/generated-inputs/coefficients/continuity6_coefficients";
  auto const pde_choice     = PDE_opts::continuity_6;
  TestType const tol_factor = std::is_same_v<double, TestType> ? 1e-14 : 1e-5;

  SECTION("level 2, degree 4")
  {
    auto const levels = fk::vector<int>{2, 2, 2, 2, 2, 2};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non uniform level: levels 2, 3, 3, 3, 2, 4, degree 4")
  {
    auto const levels = fk::vector<int>{2, 3, 3, 3, 2, 4};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_E case1 terms", "[coefficients]",
                   double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_E_case1;
  auto const gold_path  = "../testing/generated-inputs/coefficients/"
                         "fokkerplanck1_pitch_E_case1_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-13 : 1e-5;

  SECTION("level 4, degree 3")
  {
    auto const levels = fk::vector<int>{4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_E case2 terms", "[coefficients]",
                   double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_E_case2;
  auto const gold_path  = "../testing/generated-inputs/coefficients/"
                         "fokkerplanck1_pitch_E_case2_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-13 : 1e-5;

  SECTION("level 4, degree 3")
  {
    auto const levels = fk::vector<int>{4};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_pitch_C terms", "[coefficients]", double,
                   float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_pitch_C;
  auto const gold_path  = "../testing/generated-inputs/coefficients/"
                         "fokkerplanck1_pitch_C_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-5;

  SECTION("level 5, degree 2")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 2;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p3 terms", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p3;
  auto const gold_path  = "../testing/generated-inputs/coefficients/"
                         "fokkerplanck1_4p3_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;

  SECTION("level 2, degree 5")
  {
    auto const levels = fk::vector<int>{2};
    auto const degree = 5;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p4 terms", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p4;
  auto const gold_path  = "../testing/generated-inputs/coefficients/"
                         "fokkerplanck1_4p4_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-6;

  SECTION("level 5, degree 3")
  {
    auto const levels = fk::vector<int>{5};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck1_4p5 terms", "[coefficients]", double, float)
{
  auto const pde_choice = PDE_opts::fokkerplanck_1d_4p5;
  auto const gold_path  = "../testing/generated-inputs/coefficients/"
                         "fokkerplanck1_4p5_coefficients";
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;

  SECTION("level 3, degree 5")
  {
    auto const levels = fk::vector<int>{3};
    auto const degree = 5;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("fokkerplanck2_complete terms", "[coefficients]", double,
                   float)
{
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                         "fokkerplanck2_complete_coefficients";

  auto const pde_choice = PDE_opts::fokkerplanck_2d_complete;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-12 : 1e-3;

  SECTION("level 3, degree 3")
  {
    auto const levels = fk::vector<int>{3, 3};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("level 4, degree 4")
  {
    auto const levels = fk::vector<int>{4, 4};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
  SECTION("non-uniform levels: 2, 3, deg 3")
  {
    auto const levels = fk::vector<int>{2, 3};
    auto const degree = 3;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }

  SECTION("non-uniform levels: 4, 2, deg 4")
  {
    auto const levels = fk::vector<int>{4, 2};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("mirror 3 case 1 terms", "[coefficients]", double, float)
{
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                         "mirror_3_case1_coefficients";

  auto const pde_choice = PDE_opts::mirror_3_case1;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;

  SECTION("level 3, degree 4")
  {
    auto const levels = fk::vector<int>{3, 3, 3};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("mirror 3 case 2 terms", "[coefficients]", double, float)
{
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                         "mirror_3_case2_coefficients";

  auto const pde_choice = PDE_opts::mirror_3_case2;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;

  SECTION("level 3, degree 4")
  {
    auto const levels = fk::vector<int>{3, 3, 3};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}

TEMPLATE_TEST_CASE("mirror 3 case 3 terms", "[coefficients]", double, float)
{
  auto const gold_path = "../testing/generated-inputs/coefficients/"
                         "mirror_3_case3_coefficients";

  auto const pde_choice = PDE_opts::mirror_3_case3;
  TestType const tol_factor =
      std::is_same<TestType, double>::value ? 1e-14 : 1e-4;

  SECTION("level 3, degree 4")
  {
    auto const levels = fk::vector<int>{3, 3, 3};
    auto const degree = 4;
    parser const test_parse(pde_choice, levels, degree);
    test_coefficients<TestType>(test_parse, gold_path, tol_factor);
  }
}
