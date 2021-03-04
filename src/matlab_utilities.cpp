#include "matlab_utilities.hpp"
#include "tensors.hpp"
#include "tools.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

// matlab's "linspace(start, end, N)" function
//-----------------------------------------------------------------------------
//
// c++ implementation of matlab (a subset of) linspace() function
// initial c++ implementation by Tyler McDaniel
//
// -- linspace (START, END)
// -- linspace (START, END, N)
//     Return a row vector with N linearly spaced elements between START
//     and END.
//
//     If the number of elements is greater than one, then the endpoints
//     START and END are always included in the range.  If START is
//     greater than END, the elements are stored in decreasing order.  If
//     the number of points is not specified, a value of 100 is used.
//
//     The 'linspace' function returns a row vector when both START and
//     END are scalars.
//
//  (unsupported)
//     If one, or both, inputs are vectors, then
//     'linspace' transforms them to column vectors and returns a matrix
//     where each row is an independent sequence between
//     'START(ROW_N), END(ROW_N)'.
//
//     For compatibility with MATLAB, return the second argument (END)
//     when only a single value (N = 1) is requested.
//
//-----------------------------------------------------------------------------
template<typename P>
std::enable_if_t<std::is_floating_point<P>::value, fk::vector<P>>
linspace(P const start, P const end, unsigned int const num_elems)
{
  expect(num_elems > 1); // must have at least 2 elements

  // create output vector
  fk::vector<P> points(num_elems);

  // find interval size
  P const interval_size = (end - start) / (num_elems - 1);

  // insert first and last elements
  points(0)             = start;
  points(num_elems - 1) = end;

  // fill in the middle
  for (unsigned int i = 1; i < num_elems - 1; ++i)
  {
    points(i) = start + i * interval_size;
  }

  return points;
}
//-----------------------------------------------------------------------------
//
// c++ implementation of (a subset of) eye() function
// The following are not supported here:
// - providing a third "CLASS" argument
// - providing a vector argument for the dimensions
//
// -- eye (N)
// -- eye (M, N)
// -- eye ([M N])
//     Return an identity matrix.
//
//     If invoked with a single scalar argument N, return a square NxN
//     identity matrix.
//
//     If supplied two scalar arguments (M, N), 'eye' takes them to be the
//     number of rows and columns.  If given a vector with two elements,
//     'eye' uses the values of the elements as the number of rows and
//     columns, respectively.  For example:
//
//          eye (3)
//           =>  1  0  0
//               0  1  0
//               0  0  1
//
//     The following expressions all produce the same result:
//
//          eye (2)
//          ==
//          eye (2, 2)
//          ==
//          eye (size ([1, 2; 3, 4]))
//
//     Calling 'eye' with no arguments is equivalent to calling it with an
//     argument of 1.  Any negative dimensions are treated as zero.  These
//     odd definitions are for compatibility with MATLAB.
//
//-----------------------------------------------------------------------------
template<typename P>
fk::matrix<P> eye(int const M)
{
  fk::matrix<P> id(M, M);
  for (auto i = 0; i < M; ++i)
    id(i, i) = 1.0;
  return id;
}
template<typename P>
fk::matrix<P> eye(int const M, int const N)
{
  fk::matrix<P> id(M, N);
  for (auto i = 0; i < (M < N ? M : N); ++i)
    id(i, i) = 1.0;
  return id;
}

//-----------------------------------------------------------------------------
// C++ implementation of subset of matlab polyval
// Function for evaluating a polynomial.
//
// Returns the value of a polynomial p evaluated for
// x / each element of x.
// p is a vector of length n+1 whose elements are
// the coefficients of the polynomial in descending powers.

// y = p(0)*x^n + p(1)*x^(n-1) + ... + p(n-1)*x + p(n)
//-----------------------------------------------------------------------------
template<typename P>
P polyval(fk::vector<P> const p, P const x)
{
  int const num_terms = p.size();
  expect(num_terms > 0);

  P y = static_cast<P>(p(0));
  for (int i = 1; i < num_terms; ++i)
  {
    y = y * x + p(i);
  }

  return y;
}

template<typename P>
fk::vector<P> polyval(fk::vector<P> const p, fk::vector<P> const x)
{
  int const num_terms = p.size();
  int const num_sols  = x.size();
  expect(num_terms > 0);
  expect(num_sols > 0);

  fk::vector<P> solutions(num_sols);
  for (int i = 0; i < num_sols; ++i)
  {
    solutions(i) = polyval(p, x(i));
  }

  return solutions;
}

// norm( , 2) function, only for real vectors (2-norm)
template<typename P>
P l2_norm(fk::vector<P> const &vec)
{
  P accum     = 0;
  int const N = vec.size();
  for (auto i = 0; i < N; ++i)
  {
    accum += vec(i) * vec(i);
  }
  return std::sqrt(accum);
}

// norm( , 'inf') function, only for real vectors (infinity-norm)
template<typename P>
P inf_norm(fk::vector<P> const &vec)
{
  auto abs_compare = [](P const a, P const b) {
    return (std::abs(a) < std::abs(b));
  };

  P const result =
      std::abs(*std::max_element(vec.begin(), vec.end(), abs_compare));
  return result;
}

//-----------------------------------------------------------------------------
//
// these binary files can be generated from matlab or octave with
//
// function writeToFile(path, toWrite)
// fd = fopen(path,'w');
// fwrite(fd,toWrite,'double');
// fclose(fd);
// end
//
//-----------------------------------------------------------------------------
fk::vector<double> read_vector_from_bin_file(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in | std::ios::binary);

  // read failed
  if (!infile)
  {
    std::cout << "ERROR: Unable to open file " << path << "\n";
    expect(!infile.fail());
  }

  std::streampos bytes;

  // Get size, seek back to beginning
  infile.seekg(0, std::ios::end);
  bytes = infile.tellg();
  infile.seekg(0, std::ios::beg);

  // create output vector
  fk::vector<double> values;

  unsigned int const num_values = bytes / sizeof(double);
  values.resize(num_values);

  infile.read(reinterpret_cast<char *>(values.data()), bytes);

  return values;

  // infile implicitly closed on exit
}

//
// these ascii files can be generated in octave with, e.g.,
//
// w = 2
// save outfile.dat w
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
double read_scalar_from_txt_file(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed
  if (!infile)
  {
    std::cout << "ERROR: Unable to open file " << path << "\n";
    expect(!infile.fail());
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: scalar"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "type:");
  infile >> tmp_str;
  expect(tmp_str == "scalar");

  double value;

  infile >> value;

  return value;
}

//-----------------------------------------------------------------------------
//
// these ascii files can be generated in octave with, e.g.,
//
// w = linspace(-1,1);
// save outfile.dat w
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
fk::vector<double> read_vector_from_txt_file(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed, return empty
  if (!infile)
  {
    std::cout << "ERROR: Unable to open file " << path << "\n";
    expect(!infile.fail());
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: matrix"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "type:");
  infile >> tmp_str;
  expect(tmp_str == "matrix");

  // get the number of rows
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "rows:");
  infile >> tmp_str;
  int rows = std::stoi(tmp_str);

  // get the number of columns
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "columns:");
  infile >> tmp_str;
  int columns = std::stoi(tmp_str);

  // make sure we're working with a (either row or column) vector
  expect((rows == 1) || columns == 1);

  int const num_elems = (rows >= columns) ? rows : columns;

  // create output vector
  fk::vector<double> values;

  values.resize(num_elems);

  for (auto i = 0; i < num_elems; ++i)
  {
    infile >> values(i);
  }

  return values;
}

//-----------------------------------------------------------------------------
//
// these ascii files can be generated in octave with, e.g.,
//
// m = rand(3,3)
// save outfile.dat m
//
// FIXME unsure what Matlab ascii files look like
//
//-----------------------------------------------------------------------------
fk::matrix<double> read_matrix_from_txt_file(std::string const &path)
{
  // open up the file
  std::ifstream infile;
  infile.open(path, std::ios::in);

  // read failed
  if (!infile)
  {
    std::cout << "ERROR: Unable to open file " << path << "\n";
    expect(!infile.fail());
  }

  std::string tmp_str;

  getline(infile, tmp_str); // chomp the first line
  getline(infile, tmp_str); // chomp the second line

  // third line. expect "# type: matrix"
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "type:");
  infile >> tmp_str;
  expect(tmp_str == "matrix");

  // get the number of rows
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "rows:");
  infile >> tmp_str;
  int rows = std::stoi(tmp_str);

  // get the number of columns
  infile >> tmp_str; // chomp the '#'
  infile >> tmp_str;
  expect(tmp_str == "columns:");
  infile >> tmp_str;
  int columns = std::stoi(tmp_str);

  // create output matrix
  fk::matrix<double> values(rows, columns);

  for (auto i = 0; i < rows; ++i)
    for (auto j = 0; j < columns; ++j)
    {
      infile >> tmp_str;
      values(i, j) = std::stod(tmp_str);
    }

  return values;
}

// stitch matrices together side by side (all must have same # rows)
template<typename P>
fk::matrix<P> horz_matrix_concat(std::vector<fk::matrix<P>> const matrices)
{
  expect(matrices.size() > 0);
  auto const [nrows, ncols] = [&]() {
    int col_accum   = 0;
    int const nrows = matrices[0].nrows();
    for (auto const &mat : matrices)
    {
      col_accum += mat.ncols();
      expect(mat.nrows() == nrows);
    }
    return std::array<int, 2>{nrows, col_accum};
  }();
  fk::matrix<P> concat(nrows, ncols);
  int col_index = 0;
  for (auto const &mat : matrices)
  {
    concat.set_submatrix(0, col_index, mat);
    col_index += mat.ncols();
  }
  return concat;
}

//
// limited subset of matlab meshgrid
// first, creates a vector of size length.
// at each index i of this vector, the value is start + i
// the return is a size x size matrix whose rows are copies
// of this vector.
//
// the matlab version of meshgrid is more flexible, accepting
// a range as the argument and returning both the horizontal
// and vertical duplication of that range.
//
fk::matrix<int> meshgrid(int const start, int const length)
{
  expect(length > 0);
  fk::matrix<int> mesh(length, length);
  fk::vector<int> const row = [=]() {
    fk::vector<int> row(length);
    std::iota(row.begin(), row.end(), start);
    return row;
  }();
  for (int i = 0; i < mesh.nrows(); ++i)
  {
    mesh.update_row(i, row);
  }
  return mesh;
}

template<typename P, mem_type mem>
fk::matrix<P> reshape(fk::matrix<P, mem> mat, int const nrow, int const ncol)
{
  expect(nrow * ncol == mat.size());
  fk::vector<P> X(mat);
  fk::matrix<P> Xreshape(nrow, ncol);

  for (int i = 0; i < ncol; i++)
  {
    for (int j = 0; j < nrow; j++)
    {
      int const count = i * nrow + j;
      Xreshape(j, i)  = X(count);
    }
  }
  return Xreshape;
}

// explicit instantiations
#define X(T)                                                  \
  template fk::vector<T> linspace(T const start, T const end, \
                                  unsigned int const num_elems = 100);
#include "type_list_float.inc"
#undef X

#define X(T) template fk::matrix<T> eye(int const M = 1);
#include "type_list.inc"
#undef X

#define X(T) template fk::matrix<T> eye(int const M, int const N);
#include "type_list.inc"
#undef X

#define X(T) template T polyval(fk::vector<T> const p, T const x);
#include "type_list.inc"
#undef X

#define X(T) template T l2_norm(fk::vector<T> const &vec);
#include "type_list_float.inc"
#undef X

#define X(T) template T inf_norm(fk::vector<T> const &vec);
#include "type_list_float.inc"
#undef X

#define X(T) \
  template fk::vector<T> polyval(fk::vector<T> const p, fk::vector<T> const x);
#include "type_list.inc"
#undef X

#define X(T)                                 \
  template fk::matrix<T> horz_matrix_concat( \
      std::vector<fk::matrix<T>> const matrices);
#include "type_list.inc"
#undef X

#define X(T)                                                        \
  template fk::matrix<T> reshape(fk::matrix<T> mat, int const nrow, \
                                 int const ncol);
#include "type_list.inc"
#undef X
