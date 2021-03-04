#include "solver.hpp"
#include "fast_math.hpp"
#include "tools.hpp"

namespace solver
{
// simple, node-local test version
template<typename P>
P simple_gmres(fk::matrix<P> const &A, fk::vector<P> &x, fk::vector<P> const &b,
               fk::matrix<P> const &M, int const restart, int const max_iter,
               P const tolerance)
{
  expect(tolerance >= std::numeric_limits<P>::epsilon());
  expect(A.nrows() == A.ncols());
  int const n = A.nrows();
  expect(b.size() == n);
  expect(x.size() == n);

  bool const do_precond = M.size() > 0;
  std::vector<int> precond_pivots(n);
  if (do_precond)
  {
    expect(M.ncols() == n);
    expect(M.nrows() == n);
  }
  fk::matrix<P> precond(M);
  bool precond_factored = false;

  expect(restart > 0);
  expect(restart <= n);
  expect(max_iter >= restart);
  expect(max_iter <= n);

  P const norm_b = [&b]() {
    P const norm_b = fm::nrm2(b);
    return (norm_b == 0.0) ? static_cast<P>(1.0) : norm_b;
  }();

  fk::vector<P> residual(b);
  auto const compute_residual = [&A, &x, &b, &residual, &do_precond, &precond,
                                 &precond_factored, &precond_pivots]() {
    bool const trans_A = false;
    P const alpha      = -1.0;
    P const beta       = 1.0;
    residual           = b;
    fm::gemv(A, x, residual, trans_A, alpha, beta);
    if (do_precond)
    {
      precond_factored ? fm::getrs(precond, residual, precond_pivots)
                       : fm::gesv(precond, residual, precond_pivots);
      precond_factored = true;
    }
    return fm::nrm2(residual);
  };

  auto const done = [](P const error, int const outer_iters,
                       int const inner_iters) {
    std::cout << "GMRES complete with error: " << error << '\n';
    std::cout << outer_iters << " outer iterations, " << inner_iters
              << " inner iterations\n";
  };

  P const norm_r = compute_residual();
  P error        = norm_r / norm_b;
  if (error < tolerance)
  {
    done(error, 0, 0);
    return error;
  }

  fk::matrix<P> basis(n, restart + 1);
  fk::matrix<P> krylov_proj(restart + 1, restart);
  fk::vector<P> sines(restart + 1);
  fk::vector<P> cosines(restart + 1);

  int it = 0;
  int i  = 0;
  for (; it < max_iter; ++it)
  {
    P const norm_r = compute_residual();

    basis.update_col(0, residual * (1 / norm_r));

    fk::vector<P> krylov_sol(n + 1);
    krylov_sol(0) = norm_r;
    for (i = 0; i < restart; ++i)
    {
      fk::vector<P> new_basis =
          A * fk::vector<P, mem_type::view>(basis, i, 0, basis.nrows() - 1);

      if (do_precond)
      {
        fm::getrs(precond, new_basis, precond_pivots);
      }

      for (int k = 0; k <= i; ++k)
      {
        fk::vector<P, mem_type::const_view> const basis_vect(basis, k, 0,
                                                             basis.nrows() - 1);
        krylov_proj(k, i) = new_basis * basis_vect;
        new_basis         = new_basis - (basis_vect * krylov_proj(k, i));
      }
      krylov_proj(i + 1, i) = fm::nrm2(new_basis);

      basis.update_col(i + 1, new_basis * (1 / krylov_proj(i + 1, i)));
      for (int k = 0; k <= i - 1; ++k)
      {
        P const temp =
            cosines(k) * krylov_proj(k, i) + sines(k) * krylov_proj(k + 1, i);
        krylov_proj(k + 1, i) =
            -sines(k) * krylov_proj(k, i) + cosines(k) * krylov_proj(k + 1, i);
        krylov_proj(k, i) = temp;
      }

      // compute given's rotation
      lib_dispatch::rotg(krylov_proj.data(i, i), krylov_proj.data(i + 1, i),
                         cosines.data(i), sines.data(i));

      krylov_proj(i + 1, i) = 0.0;
      P const temp          = cosines(i) * krylov_sol(i);
      krylov_sol(i + 1)     = -sines(i) * krylov_sol(i);
      krylov_sol(i)         = temp;
      error                 = std::abs(krylov_sol(i + 1)) / norm_b;

      if (error <= tolerance)
      {
        auto const proj =
            fk::matrix<P, mem_type::view>(krylov_proj, 0, i, 0, i);
        std::vector<int> pivots(i + 1);

        auto s_view = fk::vector<P, mem_type::view>(krylov_sol, 0, i);
        fm::gesv(proj, s_view, pivots);
        x = x +
            (fk::matrix<P, mem_type::view>(basis, 0, basis.nrows() - 1, 0, i) *
             s_view);
        break; // depart the inner iteration loop
      }
    } // end of inner iteration loop

    if (error <= tolerance)
    {
      done(error, it, i);
      return error; // all done!
    }
    auto const proj = fk::matrix<P, mem_type::view>(krylov_proj, 0, restart - 1,
                                                    0, restart - 1);
    auto s_view     = fk::vector<P, mem_type::view>(krylov_sol, 0, restart - 1);
    std::vector<int> pivots(restart);
    fm::gesv(proj, s_view, pivots);
    x = x + (fk::matrix<P, mem_type::view>(basis, 0, basis.nrows() - 1, 0,
                                           restart - 1) *
             s_view);
    P const norm_r_outer                               = compute_residual();
    krylov_sol(std::min(krylov_sol.size() - 1, i + 1)) = norm_r_outer;
    error                                              = norm_r_outer / norm_b;

    if (error <= tolerance)
    {
      done(error, it, i);
      return error;
    }
  } // end outer iteration

  done(error, it, i);
  return error;
}

#define X(T)                                                              \
  template T simple_gmres(fk::matrix<T> const &A, fk::vector<T> &x,       \
                          fk::vector<T> const &b, fk::matrix<T> const &M, \
                          int const restart, int const max_iter,          \
                          T const tolerance);
#include "type_list_float.inc"
#undef X

} // namespace solver
