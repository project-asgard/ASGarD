#include "fast_math.hpp"
#include "tensors.hpp"
#include "tests_general.hpp"
#include <cmath>
#include <numeric>

TEMPLATE_TEST_CASE("floating point norms", "[fast_math]", float, double)
{
  /* vector 2-norm tests */
  fk::vector<TestType, mem_type::owner, resource::host> v = {
      2,  3,  4, 5, 6, 6, 5, 4, 3, 2, 10, 10, 10,
      10, 10, 7, 3, 5, 9, 9, 4, 5, 4, 5,  7};

  TestType gold_norm = 4.0 * std::sqrt(66.0);

  SECTION("fk::vector: owner, host")
  {
    TestType norm = fm::nrm2(v);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::vector: owner, device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> v_owner_dev =
        v.clone_onto_device();

    TestType norm = fm::nrm2(v_owner_dev);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::vector: const view, host")
  {
    fk::vector<TestType, mem_type::const_view, resource::host> const
        v_view_host(v);

    TestType norm = fm::nrm2(v_view_host);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::vector: const view, device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> v_owner_dev =
        v.clone_onto_device();

    fk::vector<TestType, mem_type::const_view, resource::device> const
        v_view_dev(v_owner_dev);

    TestType norm = fm::nrm2(v_view_dev);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::vector: view, host")
  {
    fk::vector<TestType, mem_type::view, resource::host> const v_view_host(v);

    TestType norm = fm::nrm2(v_view_host);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::vector: view, device")
  {
    fk::vector<TestType, mem_type::owner, resource::device> v_owner_dev =
        v.clone_onto_device();

    fk::vector<TestType, mem_type::view, resource::device> const v_view_dev(
        v_owner_dev);

    TestType norm = fm::nrm2(v_view_dev);

    relaxed_fp_comparison(gold_norm, norm);
  }

  /* matrix Frobenius norm tests */
  fk::matrix<TestType, mem_type::owner, resource::host> m = {
      {2, 3, 4, 5, 6},
      {6, 5, 4, 3, 2},
      {10, 10, 10, 10, 10},
      {7, 3, 5, 9, 9},
      {4, 5, 4, 5, 7}};

  TestType gold_norm_submatrix = 3.0 * std::sqrt(26.0);

  SECTION("fk::matrix: owner, host")
  {
    TestType norm = fm::frobenius(m);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::matrix: owner, device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> const m_owner_dev =
        m.clone_onto_device();

    TestType norm = fm::frobenius(m_owner_dev);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::matrix: view, host")
  {
    fk::matrix<TestType, mem_type::view, resource::host> m_view_host(m);

    TestType norm = fm::frobenius(m_view_host);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::matrix: view, device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> m_owner_dev =
        m.clone_onto_device();

    fk::matrix<TestType, mem_type::view, resource::device> m_view_dev(
        m_owner_dev);

    TestType norm = fm::frobenius(m_view_dev);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::matrix submatrix: view, host")
  {
    fk::matrix<TestType, mem_type::view, resource::host> const m_view(m, 2, 3,
                                                                      1, 2);

    TestType norm = fm::frobenius(m_view);

    relaxed_fp_comparison(gold_norm_submatrix, norm);
  }

  SECTION("fk::matrix submatrix: view, device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> m_owner_dev =
        m.clone_onto_device();
    fk::matrix<TestType, mem_type::view, resource::device> m_view_dev(
        m_owner_dev, 2, 3, 1, 2);

    TestType norm = fm::frobenius(m_view_dev);

    relaxed_fp_comparison(gold_norm_submatrix, norm);
  }

  SECTION("fk::matrix: const view, host")
  {
    fk::matrix<TestType, mem_type::const_view, resource::host> m_view_host(m);

    TestType norm = fm::frobenius(m_view_host);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::matrix: const view, device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> m_owner_dev =
        m.clone_onto_device();

    fk::matrix<TestType, mem_type::const_view, resource::device> const
        m_view_dev(m_owner_dev);

    TestType norm = fm::frobenius(m_view_dev);

    relaxed_fp_comparison(gold_norm, norm);
  }

  SECTION("fk::matrix submatrix: const view, host")
  {
    fk::matrix<TestType, mem_type::const_view, resource::host> const m_view(
        m, 2, 3, 1, 2);

    TestType norm = fm::frobenius(m_view);

    relaxed_fp_comparison(gold_norm_submatrix, norm);
  }

  SECTION("fk::matrix submatrix: const view, device")
  {
    fk::matrix<TestType, mem_type::owner, resource::device> m_owner_dev =
        m.clone_onto_device();
    fk::matrix<TestType, mem_type::const_view, resource::device> const
        m_view_dev(m_owner_dev, 2, 3, 1, 2);

    TestType norm = fm::frobenius(m_view_dev);

    relaxed_fp_comparison(gold_norm_submatrix, norm);
  }
}

TEST_CASE("fm::two_raised_to", "[fast_math]")
{
  SECTION("pow")
  {
    for (int i = 0; i < 31; i++)
    {
      REQUIRE(fm::two_raised_to(i) == pow(2, i));
    }
  }
}

TEMPLATE_TEST_CASE("fm::gemm", "[fast_math]", float, double, int)
{
  // clang-format off
    fk::matrix<TestType> const ans{
        {360,  610,  860},
        {710, 1210, 1710},
    };

    fk::matrix<TestType> const in1{
        {3, 4,  5,  6,  7},
        {8, 9, 10, 11, 12},
    }; 
    fk::matrix<TestType> const in2{
        {12, 22, 32},
	{13, 23, 33},
	{14, 24, 34},
	{15, 25, 35},
	{16, 26, 36},
    };
  // clang-format on

  fk::matrix<TestType, mem_type::owner, resource::device> const in1_d(
      in1.clone_onto_device());
  fk::matrix<TestType, mem_type::owner, resource::device> const in2_d(
      in2.clone_onto_device());

  SECTION("no transpose")
  {
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    fm::gemm(in1, in2, result);
    REQUIRE(result == ans);
  }
  SECTION("no transpose, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          in1.nrows(), in2.ncols());
      fm::gemm(in1_d, in2_d, result_d);
      fk::matrix<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }

  SECTION("transpose a")
  {
    fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    bool const trans_A = true;
    fm::gemm(in1_t, in2, result, trans_A);
    REQUIRE(result == ans);
  }

  SECTION("transpose a, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
      fk::matrix<TestType, mem_type::owner, resource::device> const in1_t_d(
          in1_t.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          in1.nrows(), in2.ncols());
      bool const trans_A = true;
      fm::gemm(in1_t_d, in2_d, result_d, trans_A);
      fk::matrix<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }

  SECTION("transpose b")
  {
    fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    bool const trans_A = false;
    bool const trans_B = true;
    fm::gemm(in1, in2_t, result, trans_A, trans_B);
    REQUIRE(result == ans);
  }

  SECTION("transpose b, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
      fk::matrix<TestType, mem_type::owner, resource::device> const in2_t_d(
          in2_t.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          in1.nrows(), in2.ncols());
      bool const trans_A = false;
      bool const trans_B = true;
      fm::gemm(in1_d, in2_t_d, result_d, trans_A, trans_B);
      fk::matrix<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }

  SECTION("both transpose")
  {
    fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
    fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    bool const trans_A = true;
    bool const trans_B = true;
    fm::gemm(in1_t, in2_t, result, trans_A, trans_B);
    REQUIRE(result == ans);
  }

  SECTION("both transpose, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> const in1_t = fk::matrix<TestType>(in1).transpose();
      fk::matrix<TestType> const in2_t = fk::matrix<TestType>(in2).transpose();
      fk::matrix<TestType, mem_type::owner, resource::device> const in1_t_d(
          in1_t.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> const in2_t_d(
          in2_t.clone_onto_device());
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          in1.nrows(), in2.ncols());
      bool const trans_A = true;
      bool const trans_B = true;
      fm::gemm(in1_t_d, in2_t_d, result_d, trans_A, trans_B);
      fk::matrix<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }

  SECTION("test scaling")
  {
    fk::matrix<TestType> result(in1.nrows(), in2.ncols());
    std::fill(result.begin(), result.end(), 1.0);

    fk::matrix<TestType> const gold = [&] {
      fk::matrix<TestType> ans = (in1 * in2) * 2.0;
      std::transform(
          ans.begin(), ans.end(), ans.begin(),
          [](TestType const elem) -> TestType { return elem + 1.0; });

      return ans;
    }();

    bool const trans_A   = false;
    bool const trans_B   = false;
    TestType const alpha = 2.0;
    TestType const beta  = 1.0;

    fm::gemm(in1, in2, result, trans_A, trans_B, alpha, beta);
    REQUIRE(result == gold);
  }

  SECTION("test scaling, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> result(in1.nrows(), in2.ncols());
      std::fill(result.begin(), result.end(), 1.0);
      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          result.clone_onto_device());

      fk::matrix<TestType> const gold = [&] {
        fk::matrix<TestType> ans = (in1 * in2) * 2.0;
        std::transform(
            ans.begin(), ans.end(), ans.begin(),
            [](TestType const elem) -> TestType { return elem + 1.0; });

        return ans;
      }();

      bool const trans_A   = false;
      bool const trans_B   = false;
      TestType const alpha = 2.0;
      TestType const beta  = 1.0;

      fm::gemm(in1_d, in2_d, result_d, trans_A, trans_B, alpha, beta);
      result.transfer_from(result_d);
      REQUIRE(result == gold);
    }
  }

  // test the case where lda doesn't equal the number of rows in
  // arguments - this often occurs when using views.
  SECTION("lda =/= nrows")
  {
    fk::matrix<TestType> in1_extended(in1.nrows() + 1, in1.ncols());

    fk::matrix<TestType, mem_type::view> in1_view(
        in1_extended, 0, in1.nrows() - 1, 0, in1.ncols() - 1);
    in1_view = in1;

    fk::matrix<TestType> in2_extended(in2.nrows() + 1, in2.ncols());

    fk::matrix<TestType, mem_type::view> in2_view(
        in2_extended, 0, in2.nrows() - 1, 0, in2.ncols() - 1);
    in2_view = in2;

    fk::matrix<TestType> result(in1.nrows() + 2, in2.ncols());
    fk::matrix<TestType, mem_type::view> result_view(result, 0, in1.nrows() - 1,
                                                     0, in2.ncols() - 1);

    fm::gemm(in1_view, in2_view, result_view);
    REQUIRE(result_view == ans);
  }

  SECTION("lda =/= nrows, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType, mem_type::owner, resource::device> in1_extended_d(
          in1.nrows() + 1, in1.ncols());

      fk::matrix<TestType, mem_type::view, resource::device> in1_view_d(
          in1_extended_d, 0, in1.nrows() - 1, 0, in1.ncols() - 1);
      in1_view_d.transfer_from(in1);

      fk::matrix<TestType, mem_type::owner, resource::device> in2_extended_d(
          in2.nrows() + 1, in2.ncols());

      fk::matrix<TestType, mem_type::view, resource::device> in2_view_d(
          in2_extended_d, 0, in2.nrows() - 1, 0, in2.ncols() - 1);
      in2_view_d.transfer_from(in2);

      fk::matrix<TestType, mem_type::owner, resource::device> result_d(
          in1.nrows() + 2, in2.ncols());
      fk::matrix<TestType, mem_type::view, resource::device> result_view_d(
          result_d, 0, in1.nrows() - 1, 0, in2.ncols() - 1);

      fm::gemm(in1_view_d, in2_view_d, result_view_d);
      fk::matrix<TestType> const result(result_view_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }
}

TEMPLATE_TEST_CASE("fm::gemv", "[fast_math]", float, double, int)
{
  // clang-format off
    fk::vector<TestType> const ans
      {27, 42, 69, 71, -65};

    fk::matrix<TestType> const A {
     {-10, -2, -7},
     {  6, -5, -5},
     {  7,  6, -7},
     {  8, -1, -8},
     { -9,  9,  8},
    };

    fk::vector<TestType> const x
     {2, 1, -7};
  // clang-format on

  fk::matrix<TestType, mem_type::owner, resource::device> const A_d(
      A.clone_onto_device());
  fk::vector<TestType, mem_type::owner, resource::device> const x_d(
      x.clone_onto_device());

  SECTION("no transpose")
  {
    fk::vector<TestType> result(ans.size());
    fm::gemv(A, x, result);
    REQUIRE(result == ans);
  }

  SECTION("no transpose, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> result_d(
          ans.size());
      fm::gemv(A_d, x_d, result_d);
      fk::vector<TestType> const result(result_d.clone_onto_host());
      REQUIRE(result == ans);
    }
  }
  SECTION("transpose A")
  {
    fk::matrix<TestType> const A_trans = fk::matrix<TestType>(A).transpose();
    fk::vector<TestType> result(ans.size());
    bool const trans_A = true;
    fm::gemv(A_trans, x, result, trans_A);
    REQUIRE(result == ans);
  }

  SECTION("transpose A, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::matrix<TestType> const A_trans = fk::matrix<TestType>(A).transpose();
      fk::matrix<TestType, mem_type::owner, resource::device> const A_trans_d(
          A_trans.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> result_d(
          ans.size());
      bool const trans_A = true;
      fm::gemv(A_trans_d, x_d, result_d, trans_A);
      fk::vector<TestType> const result(result_d.clone_onto_host());

      REQUIRE(result == ans);
    }
  }

  SECTION("test scaling")
  {
    fk::vector<TestType> result(ans.size());
    std::fill(result.begin(), result.end(), 1.0);

    fk::vector<TestType> const gold = [&] {
      fk::vector<TestType> ans = (A * x) * 2.0;
      std::transform(
          ans.begin(), ans.end(), ans.begin(),
          [](TestType const elem) -> TestType { return elem + 1.0; });

      return ans;
    }();

    bool const trans_A   = false;
    TestType const alpha = 2.0;
    TestType const beta  = 1.0;

    fm::gemv(A, x, result, trans_A, alpha, beta);
    REQUIRE(result == gold);
  }

  SECTION("test scaling, device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType> result(ans.size());
      std::fill(result.begin(), result.end(), 1.0);
      fk::vector<TestType, mem_type::owner, resource::device> result_d(
          result.clone_onto_device());

      fk::vector<TestType> const gold = [&] {
        fk::vector<TestType> ans = (A * x) * 2.0;
        std::transform(
            ans.begin(), ans.end(), ans.begin(),
            [](TestType const elem) -> TestType { return elem + 1.0; });

        return ans;
      }();

      bool const trans_A   = false;
      TestType const alpha = 2.0;
      TestType const beta  = 1.0;

      fm::gemv(A_d, x_d, result_d, trans_A, alpha, beta);
      result.transfer_from(result_d);
      REQUIRE(result == gold);
    }
  }
}

TEMPLATE_TEST_CASE("other vector routines", "[fast_math]", float, double, int)
{
  fk::vector<TestType> const gold = {2, 3, 4, 5, 6};
  SECTION("vector scale and accumulate (fm::axpy)")
  {
    TestType const scale = 2.0;

    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> rhs{7, 8, 9, 10, 11};
    fk::vector<TestType> rhs_own(rhs);
    fk::vector<TestType, mem_type::view> rhs_view(rhs_own);

    fk::vector<TestType> const ans = {16, 19, 22, 25, 28};

    REQUIRE(fm::axpy(rhs, test, scale) == ans);
    test = gold;
    REQUIRE(fm::axpy(rhs_view, test, scale) == ans);

    REQUIRE(fm::axpy(rhs, test_view, scale) == ans);
    REQUIRE(test_own == ans);
    test_view = gold;
    REQUIRE(fm::axpy(rhs_view, test_view, scale) == ans);
    REQUIRE(test_own == ans);
  }

  SECTION("vector scale and accumulate (fm::axpy), device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      TestType const scale = 2.0;

      fk::vector<TestType, mem_type::owner, resource::device> test_d(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> test_own_d(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::view, resource::device> test_view_d(
          test_own_d);

      fk::vector<TestType, mem_type::owner, resource::device> rhs_d{7, 8, 9, 10,
                                                                    11};
      fk::vector<TestType, mem_type::owner, resource::device> rhs_own_d(rhs_d);
      fk::vector<TestType, mem_type::view, resource::device> rhs_view_d(
          rhs_own_d);

      fk::vector<TestType> const ans = {16, 19, 22, 25, 28};
      fk::vector<TestType> result(ans.size());

      result.transfer_from(fm::axpy(rhs_d, test_d, scale));
      REQUIRE(result == ans);

      test_d.transfer_from(gold);

      result.transfer_from(fm::axpy(rhs_view_d, test_d, scale));
      REQUIRE(result == ans);

      result.transfer_from(fm::axpy(rhs_d, test_view_d, scale));
      REQUIRE(result == ans);
      result.transfer_from(test_own_d);
      REQUIRE(result == ans);
      test_view_d.transfer_from(gold);
      result.transfer_from(fm::axpy(rhs_view_d, test_view_d, scale));
      REQUIRE(result == ans);
      result.transfer_from(test_own_d);
      REQUIRE(result == ans);
    }
  }

  SECTION("vector copy (fm::copy)")
  {
    fk::vector<TestType> test(gold.size());
    fk::vector<TestType> test_own(gold.size());
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType, mem_type::const_view> const gold_view(gold);

    REQUIRE(fm::copy(gold, test) == gold);
    test.scale(0);
    REQUIRE(fm::copy(gold_view, test) == gold);

    REQUIRE(fm::copy(gold, test_view) == gold);
    REQUIRE(test_own == gold);
    test_own.scale(0);
    REQUIRE(fm::copy(gold_view, test_view) == gold);
    REQUIRE(test_own == gold);
  }

  SECTION("vector copy (fm::copy), device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      fk::vector<TestType, mem_type::owner, resource::device> test_d(
          gold.size());
      fk::vector<TestType, mem_type::owner, resource::device> test_own_d(
          gold.size());
      fk::vector<TestType, mem_type::view, resource::device> test_view_d(
          test_own_d);

      fk::vector<TestType, mem_type::owner, resource::device> const gold_d(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::const_view, resource::device> const
          gold_view_d(gold_d);

      fk::vector<TestType> answer(gold.size());

      answer.transfer_from(fm::copy(gold_d, test_d));
      REQUIRE(answer == gold);
      fm::scal(static_cast<TestType>(0.0), test_d);
      answer.transfer_from(fm::copy(gold_view_d, test_d));
      REQUIRE(answer == gold);

      answer.transfer_from(fm::copy(gold_d, test_view_d));
      REQUIRE(answer == gold);
      answer.transfer_from(test_own_d);
      REQUIRE(answer == gold);

      fm::scal(static_cast<TestType>(0.0), test_own_d);
      answer.transfer_from(fm::copy(gold_view_d, test_view_d));
      REQUIRE(answer == gold);
      answer.transfer_from(test_own_d);
      REQUIRE(answer == gold);
    }
  }

  SECTION("vector scale (fm::scal)")
  {
    TestType const x = 2.0;
    fk::vector<TestType> test(gold);
    fk::vector<TestType> test_own(gold);
    fk::vector<TestType, mem_type::view> test_view(test_own);

    fk::vector<TestType> const ans = {4, 6, 8, 10, 12};

    REQUIRE(fm::scal(x, test) == ans);
    REQUIRE(fm::scal(x, test_view) == ans);
    REQUIRE(test_own == ans);

    test     = gold;
    test_own = gold;

    TestType const x2 = 0.0;
    fk::vector<TestType> const zeros(gold.size());

    REQUIRE(fm::scal(x2, test) == zeros);
    REQUIRE(fm::scal(x2, test_view) == zeros);
    REQUIRE(test_own == zeros);
  }

  SECTION("vector scale (fm::scal), device")
  {
    if constexpr (std::is_floating_point_v<TestType>)
    {
      TestType const x = 2.0;
      fk::vector<TestType, mem_type::owner, resource::device> test_d(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::owner, resource::device> test_own_d(
          gold.clone_onto_device());
      fk::vector<TestType, mem_type::view, resource::device> test_view_d(
          test_own_d);

      fk::vector<TestType> const ans = {4, 6, 8, 10, 12};

      fk::vector<TestType> result(ans.size());

      result.transfer_from(fm::scal(x, test_d));
      REQUIRE(result == ans);

      result.transfer_from(fm::scal(x, test_view_d));
      REQUIRE(result == ans);
      result.transfer_from(test_own_d);
      REQUIRE(result == ans);

      test_d.transfer_from(gold);
      test_own_d.transfer_from(gold);

      TestType const x2 = 0.0;
      fk::vector<TestType> const zeros(gold.size());

      result.transfer_from(fm::scal(x2, test_d));
      REQUIRE(result == zeros);

      result.transfer_from(fm::scal(x2, test_view_d));
      REQUIRE(result == zeros);
      result.transfer_from(test_own_d);
      REQUIRE(result == zeros);
    }
  }
}

TEMPLATE_TEST_CASE("LU Routines", "[fast_math]", float, double)
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

  SECTION("gesv and getrs")
  {
    fk::matrix<TestType> const A_copy = A_gold;
    std::vector<int> ipiv(A_copy.nrows());
    fk::vector<TestType> x = B_gold;

    fm::gesv(A_copy, x, ipiv);

    TestType const tol_factor =
        std::is_same<TestType, double>::value ? 1e-16 : 1e-7;

    rmse_comparison(A_copy, LU_gold, tol_factor);
    rmse_comparison(x, X_gold, tol_factor);
    x = B1_gold;
    fm::getrs(A_copy, x, ipiv);
    rmse_comparison(x, X1_gold, tol_factor);
  }
}
