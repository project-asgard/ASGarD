#pragma once
#include "build_info.hpp"

#ifdef ASGARD_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "lib_dispatch.hpp"
#include "tools.hpp"

#include <memory>
#include <string>
#include <vector>

/* tolerance for answer comparisons */
#define TOL std::numeric_limits<P>::epsilon() * 2

// allows a private member function to declare via its parameter list who from
// outside the class is allowed to call it. you must hold an "access badge".
template<typename badge_holder>
class access_badge
{
  friend badge_holder;
  access_badge(){};
};

// used to suppress warnings in unused variables
auto const ignore = [](auto ignored) { (void)ignored; };

enum class mem_type
{
  owner,
  view,
  const_view
};

template<mem_type mem>
using enable_for_owner = std::enable_if_t<mem == mem_type::owner>;

template<mem_type mem>
using enable_for_all_views =
    std::enable_if_t<mem == mem_type::view || mem == mem_type::const_view>;

// enable only for const views
template<mem_type mem>
using enable_for_const_view = std::enable_if_t<mem == mem_type::const_view>;

// enable only for nonconst views
template<mem_type mem>
using enable_for_view = std::enable_if_t<mem == mem_type::view>;

// disable for const views
template<mem_type mem>
using disable_for_const_view =
    std::enable_if_t<mem == mem_type::owner || mem == mem_type::view>;

template<resource resrc>
using enable_for_host = std::enable_if_t<resrc == resource::host>;

template<resource resrc>
using enable_for_device = std::enable_if_t<resrc == resource::device>;

// resource arguments allow developers to select host (CPU only) or device
// (accelerator if enabled, fall back to host) allocation for tensors. device
// tensors have a restricted API - most member functions are disabled. the fast
// math component is designed to allow BLAS on host and device tensors.

// mem_type arguments allow for the selection of owner or view (read/write
// window into underlying owner memory) semantics.

// device owners can be constructed with no-arg, size, or
// initializer list constructors.
//
// device owners are allocated in accelerator DRAM when
// the appropriate build option is set, with allocation
// falling back to CPU RAM otherwise.
//
// additionally, device owners can be transfer constructed from a
// host owner or copy/move constructed from another device owner.
//
// host owners can be created with any of the below constructors.
//
// device views can only be created from a device owner, and host
// views can only be constructor from host owners

namespace fk
{
// forward declarations
template<typename P, mem_type mem = mem_type::owner,
         resource resrc = resource::host> // default to be an owner only on host
class vector;
template<typename P, mem_type mem = mem_type::owner,
         resource resrc = resource::host>
class matrix;

template<typename P, mem_type mem, resource resrc>
class vector
{
  // all types of vectors are mutual friends
  template<typename, mem_type, resource>
  friend class vector;

public:
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector();
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  explicit vector(int const size);
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector(std::initializer_list<P> list);
  template<mem_type m_ = mem, typename = enable_for_owner<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  vector(std::vector<P> const &);

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector(fk::matrix<P, mem_type::owner, resrc> const &);

  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit vector(fk::vector<P, omem, resrc> &vec, int const start_index,
                  int const stop_index);
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit vector(fk::vector<P, omem, resrc> const &vec, int const start_index,
                  int const stop_index);

  // overloads for default case - whole vector
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit vector(fk::vector<P, omem, resrc> &owner);

  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit vector(fk::vector<P, omem, resrc> const &owner);

  // create vector view from matrix
  // const view version
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit vector(fk::matrix<P, omem, resrc> const &source, int const col_index,
                  int const row_start, int const row_stop);
  // modifiable view version
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit vector(fk::matrix<P, omem, resrc> &source, int const col_index,
                  int const row_start, int const row_stop);

  ~vector();

  // constructor/assignment (required to be same to same T==T)
  vector(vector<P, mem, resrc> const &);
  // cannot be templated per C++ spec 12.8
  // instead of disabling w/ sfinae for const_view,
  // static assert added to definition
  vector<P, mem, resrc> &operator=(vector<P, mem, resrc> const &);

  // move constructor/assignment (required to be same to same)
  vector(vector<P, mem, resrc> &&);

  // as with copy assignment, static assert added
  // to definition to prevent assignment into
  // const views
  vector<P, mem, resrc> &operator=(vector<P, mem, resrc> &&);

  // copy construct owner from view values
  template<mem_type omem, mem_type m_ = mem, typename = enable_for_owner<m_>,
           mem_type m__ = omem, typename = enable_for_all_views<m__>>
  explicit vector(vector<P, omem, resrc> const &);

  // assignment owner <-> view
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>>
  vector<P, mem, resrc> &operator=(vector<P, omem, resrc> const &);

  // converting constructor/assignment overloads
  template<typename PP, mem_type omem, mem_type m_ = mem,
           typename = enable_for_owner<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  explicit vector(vector<PP, omem> const &);
  template<typename PP, mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  vector<P, mem> &operator=(vector<PP, omem> const &);

  // device transfer
  // host to device, new vector
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P, mem_type::owner, resource::device> clone_onto_device() const;
  // host to device copy
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_device<r_>>
  vector<P, mem, resrc> &transfer_from(vector<P, omem, resource::host> const &);
  // device to host, new vector
  template<resource r_ = resrc, typename = enable_for_device<r_>>
  vector<P, mem_type::owner, resource::host> clone_onto_host() const;
  // device to host copy
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  vector<P, mem, resrc> &
  transfer_from(vector<P, omem, resource::device> const &);

  //
  // copy out of std::vector
  //
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P, mem> &operator=(std::vector<P> const &);

  //
  // copy into std::vector
  //
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  std::vector<P> to_std() const;

  //
  // subscripting operators
  //
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  P &operator()(int const);
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  P operator()(int const) const;

  //
  // comparison operators
  //
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator==(vector<P, omem> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator!=(vector<P, omem> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator<(vector<P, omem> const &) const;

  //
  // math operators
  //
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator+(vector<P, omem> const &right) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator-(vector<P, omem> const &right) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  P operator*(vector<P, omem> const &)const;

  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator*(matrix<P, omem> const &)const;

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator*(P const) const;

  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> single_column_kron(vector<P, omem> const &) const;

  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P, mem> &scale(P const x);

  //
  // basic queries to private data
  //
  int size() const { return size_; }

  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  // this can be offsetted for views
  P *data(int const elem = 0) const { return &data_[elem]; }

  // this is to allow specific other types to access the private ref counter of
  // owners - specifically, we want to allow a matrix<view> to be made from a
  // vector<owner/view>
  std::shared_ptr<int>
  get_ref_count(access_badge<matrix<P, mem_type::view, resrc>> const)
  {
    return ref_count_;
  }

  std::shared_ptr<int> get_ref_count(
      access_badge<matrix<P, mem_type::const_view, resrc>> const) const
  {
    return ref_count_;
  }

  //
  // utility functions
  //
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void print(std::string const label = "") const;

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void dump_to_octave(char const *) const;

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  fk::vector<P, mem_type::owner, resrc> &resize(int const size = 0);

  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  vector<P, mem> &set_subvector(int const, vector<P, omem> const);

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> extract(int const, int const) const;

  template<mem_type omem, mem_type m_ = mem, typename = enable_for_owner<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P, mem> &concat(vector<P, omem> const &right);

  typedef P *iterator;
  typedef const P *const_iterator;

  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  iterator begin()
  {
    return data();
  }
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  iterator end()
  {
    return data() + size();
  }

  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + size(); }

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  int get_num_views() const;

private:
  // const/nonconst view constructors delegate to this private constructor
  // delegated is a dummy variable to enable resolution
  template<mem_type m_ = mem, typename = enable_for_all_views<m_>,
           mem_type omem>
  explicit vector(fk::vector<P, omem, resrc> const &vec, int const start_index,
                  int const stop_index, bool const delegated);

  // vector view from matrix constructors (both const/nonconst) delegate
  // to this private constructor, also with a dummy variable
  template<mem_type omem, mem_type m_ = mem,
           typename = enable_for_all_views<m_>>
  explicit vector(fk::matrix<P, omem, resrc> const &source,
                  std::shared_ptr<int> source_ref_count, int const column_index,
                  int const row_start, int const row_stop);

  P *data_;  //< pointer to elements
  int size_; //< dimension
  std::shared_ptr<int> ref_count_ = nullptr;
};

template<typename P, mem_type mem, resource resrc>
class matrix
{
  template<typename, mem_type, resource>
  friend class matrix; // so that views can access owner sharedptr/rows

  // template on pointer/ref type to get iterator and const iterator
  template<typename T, typename R>
  class matrix_iterator; // forward declaration for custom iterator; defined
                         // out of line

public:
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix();
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix(int rows, int cols);
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix(std::initializer_list<std::initializer_list<P>> list);

  // create const view
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit matrix(fk::matrix<P, omem, resrc> const &owner, int const start_row,
                  int const stop_row, int const start_col, int const stop_col);
  // create modifiable view
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit matrix(fk::matrix<P, omem, resrc> &owner, int const start_row,
                  int const stop_row, int const start_col, int const stop_col);

  // overloads for default case - whole matrix
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit matrix(fk::matrix<P, omem, resrc> const &owner);

  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit matrix(fk::matrix<P, omem, resrc> &owner);

  // create matrix view from vector
  // const view version
  template<mem_type m_ = mem, typename = enable_for_const_view<m_>,
           mem_type omem>
  explicit matrix(fk::vector<P, omem, resrc> const &source, int const num_rows,
                  int const num_cols, int const start_index = 0);
  // modifiable view version
  template<mem_type m_ = mem, typename = enable_for_view<m_>, mem_type omem,
           mem_type om_ = omem, typename = disable_for_const_view<om_>>
  explicit matrix(fk::vector<P, omem, resrc> &source, int const num_rows,
                  int const num_cols, int const start_index = 0);

  ~matrix();

  // copy constructor/assign
  matrix(matrix<P, mem, resrc> const &);
  matrix<P, mem, resrc> &operator=(matrix<P, mem, resrc> const &);

  // copy construct owner from view values
  template<mem_type omem, mem_type m_ = mem, typename = enable_for_owner<m_>,
           mem_type m__ = omem, typename = enable_for_all_views<m__>>
  explicit matrix(matrix<P, omem, resrc> const &);

  // assignment owner <-> view
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>>
  matrix<P, mem, resrc> &operator=(matrix<P, omem, resrc> const &);

  // converting construction/assign
  template<typename PP, mem_type omem, mem_type m_ = mem,
           typename = enable_for_owner<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  explicit matrix(matrix<PP, omem> const &);

  template<typename PP, mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &operator=(matrix<PP, omem> const &);

  // host to device, new matrix
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  fk::matrix<P, mem_type::owner, resource::device> clone_onto_device() const;
  // host to device copy
  template<mem_type omem, resource r_ = resrc, typename = enable_for_device<r_>>
  matrix<P, mem, resrc> &transfer_from(matrix<P, omem, resource::host> const &);
  // device to host, new matrix
  template<resource r_ = resrc, typename = enable_for_device<r_>>
  fk::matrix<P, mem_type::owner, resource::host> clone_onto_host() const;
  // device to host copy
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem, resrc> &
  transfer_from(matrix<P, omem, resource::device> const &);

  // move constructor/assign
  matrix(matrix<P, mem, resrc> &&);
  matrix<P, mem, resrc> &operator=(matrix<P, mem, resrc> &&);

  //
  // copy out of fk::vector
  //
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &operator=(fk::vector<P, omem> const &);

  //
  // subscripting operators
  //
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  P &operator()(int const, int const);

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  P operator()(int const, int const) const;

  //
  // comparison operators
  //
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator==(matrix<P, omem> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator!=(matrix<P, omem> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  bool operator<(matrix<P, omem> const &) const;

  //
  // math operators
  //
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> operator*(P const) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  vector<P> operator*(vector<P, omem> const &)const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> operator*(matrix<P, omem> const &)const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> operator+(matrix<P, omem> const &) const;
  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> operator-(matrix<P, omem> const &) const;

  template<mem_type m_ = mem, typename = enable_for_owner<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem, resrc> &transpose();

  template<mem_type m_ = mem, typename = enable_for_owner<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem, resrc> &ip_transpose();

  template<mem_type omem, resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> kron(matrix<P, omem> const &) const;

  template<typename U  = P,
           typename    = std::enable_if_t<std::is_floating_point<U>::value &&
                                       std::is_same<P, U>::value>,
           mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem> &invert();

  template<typename U  = P,
           typename    = std::enable_if_t<std::is_floating_point<U>::value &&
                                       std::is_same<P, U>::value>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  P determinant() const;

  //
  // basic queries to private data
  //
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  // for owners: stride == nrows
  // for views:  stride == owner's nrows
  int stride() const { return stride_; }
  int size() const { return nrows() * ncols(); }
  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  P *data(int const i = 0, int const j = 0) const
  {
    // return &data_[i * stride() + j]; // row-major
    return &data_[j * stride() + i]; // column-major
  }

  //
  // utility functions
  //
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &update_col(int const, fk::vector<P, omem> const &);
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem> &update_col(int const, std::vector<P> const &);
  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &update_row(int const, fk::vector<P, omem> const &);
  template<mem_type m_ = mem, typename = disable_for_const_view<m_>,
           resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P, mem> &update_row(int const, std::vector<P> const &);

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  matrix<P, mem_type::owner, resrc> &clear_and_resize(int const, int const);

  template<mem_type omem, mem_type m_ = mem,
           typename = disable_for_const_view<m_>, resource r_ = resrc,
           typename = enable_for_host<r_>>
  matrix<P, mem> &set_submatrix(int const row_idx, int const col_idx,
                                fk::matrix<P, omem> const &submatrix);
  template<resource r_ = resrc, typename = enable_for_host<r_>>
  matrix<P> extract_submatrix(int const row_idx, int const col_idx,
                              int const num_rows, int const num_cols) const;

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void print(std::string const label = "") const;

  template<resource r_ = resrc, typename = enable_for_host<r_>>
  void dump_to_octave(char const *name) const;

  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  int get_num_views() const;

  // this is to allow specific other types to access the private ref counter of
  // owners - specifically, we want to allow a vector<view> to be made from a
  // matrix<owner/view>
  std::shared_ptr<int>
  get_ref_count(access_badge<vector<P, mem_type::view, resrc>> const)
  {
    return ref_count_;
  }

  std::shared_ptr<int> get_ref_count(
      access_badge<vector<P, mem_type::const_view, resrc>> const) const
  {
    return ref_count_;
  }

  using iterator       = matrix_iterator<P *, P &>;
  using const_iterator = matrix_iterator<P const *, P const &>;

  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  iterator begin()
  {
    return iterator(data(), stride(), nrows());
  }

  template<mem_type m_ = mem, typename = disable_for_const_view<m_>>
  iterator end()
  {
    return iterator(data() + stride() * ncols(), stride(), nrows());
  }

  const_iterator begin() const
  {
    return const_iterator(data(), stride(), nrows());
  }

  const_iterator end() const
  {
    return const_iterator(data() + stride() * ncols(), stride(), nrows());
  }

private:
  // matrix view constructors (both const/nonconst) delegate to this private
  // constructor delegated is a dummy variable to assist in overload resolution
  template<mem_type m_ = mem, typename = enable_for_all_views<m_>,
           mem_type omem>
  explicit matrix(fk::matrix<P, omem, resrc> const &owner, int const start_row,
                  int const stop_row, int const start_col, int const stop_col,
                  bool const delegated);

  // matrix view from vector owner constructors (both const/nonconst) delegate
  // to this private constructor, also with a dummy variable
  template<mem_type omem, mem_type m_ = mem,
           typename = enable_for_all_views<m_>>
  explicit matrix(fk::vector<P, omem, resrc> const &source,
                  std::shared_ptr<int> source_ref_count, int const num_rows,
                  int const num_cols, int const start_index);

  P *data_;    //< pointer to elements
  int nrows_;  //< row dimension
  int ncols_;  //< column dimension
  int stride_; //< leading dimension;
               // number of elements in memory between successive matrix
               // elements in a row
  std::shared_ptr<int> ref_count_ = nullptr;
};

//-----------------------------------------------------------------------------
//
// device allocation and transfer helpers
//
//-----------------------------------------------------------------------------

template<typename P>
inline void
allocate_device(P *&ptr, int const num_elems, bool const initialize = true)
{
#ifdef ASGARD_USE_CUDA
  auto success = cudaMalloc((void **)&ptr, num_elems * sizeof(P));
  assert(success == cudaSuccess);
  if (num_elems > 0)
  {
    expect(ptr != nullptr);
  }

  if (initialize)
  {
    success = cudaMemset((void *)ptr, 0, num_elems * sizeof(P));
    expect(success == cudaSuccess);
  }

#else
  if (initialize)
  {
    ptr = new P[num_elems]();
  }
  else
  {
    ptr = new P[num_elems];
  }
#endif
}

template<typename P>
inline void delete_device(P *const ptr)
{
#ifdef ASGARD_USE_CUDA
  auto const success = cudaFree(ptr);
  // the device runtime may be unloaded at process shut down
  // (when static storage duration destructors are called)
  // returning a cudartUnloading error code.
  expect((success == cudaSuccess) || (success == cudaErrorCudartUnloading));
#else
  delete[] ptr;
#endif
}

template<typename P>
inline void
copy_on_device(P *const dest, P const *const source, int const num_elems)
{
#ifdef ASGARD_USE_CUDA
  auto const success =
      cudaMemcpy(dest, source, num_elems * sizeof(P), cudaMemcpyDeviceToDevice);
  expect(success == cudaSuccess);
#else
  std::copy(source, source + num_elems, dest);
#endif
}

template<typename P>
inline void
copy_to_device(P *const dest, P const *const source, int const num_elems)
{
#ifdef ASGARD_USE_CUDA
  auto const success =
      cudaMemcpy(dest, source, num_elems * sizeof(P), cudaMemcpyHostToDevice);
  expect(success == cudaSuccess);
#else
  std::copy(source, source + num_elems, dest);
#endif
}

template<typename P>
inline void
copy_to_host(P *const dest, P const *const source, int const num_elems)
{
#ifdef ASGARD_USE_CUDA
  auto const success =
      cudaMemcpy(dest, source, num_elems * sizeof(P), cudaMemcpyDeviceToHost);
  expect(success == cudaSuccess);
#else
  std::copy(source, source + num_elems, dest);
#endif
}

template<typename P, mem_type mem, mem_type omem>
inline void
copy_matrix_on_device(fk::matrix<P, mem, resource::device> &dest,
                      fk::matrix<P, omem, resource::device> const &source)
{
  expect(source.nrows() == dest.nrows());
  expect(source.ncols() == dest.ncols());

#ifdef ASGARD_USE_CUDA
  auto const success =
      cudaMemcpy2D(dest.data(), dest.stride() * sizeof(P), source.data(),
                   source.stride() * sizeof(P), source.nrows() * sizeof(P),
                   source.ncols(), cudaMemcpyDeviceToDevice);
  expect(success == 0);
#else
  std::copy(source.begin(), source.end(), dest.begin());
#endif
}

template<typename P, mem_type mem, mem_type omem, mem_type m_ = mem,
         typename = disable_for_const_view<m_>>
inline void
copy_matrix_to_device(fk::matrix<P, mem, resource::device> &dest,
                      fk::matrix<P, omem, resource::host> const &source)
{
  expect(source.nrows() == dest.nrows());
  expect(source.ncols() == dest.ncols());
#ifdef ASGARD_USE_CUDA
  auto const success =
      cudaMemcpy2D(dest.data(), dest.stride() * sizeof(P), source.data(),
                   source.stride() * sizeof(P), source.nrows() * sizeof(P),
                   source.ncols(), cudaMemcpyHostToDevice);
  expect(success == 0);
#else
  std::copy(source.begin(), source.end(), dest.begin());
#endif
}

template<typename P, mem_type mem, mem_type omem, mem_type m_ = mem,
         typename = disable_for_const_view<m_>>
inline void
copy_matrix_to_host(fk::matrix<P, mem, resource::host> &dest,
                    fk::matrix<P, omem, resource::device> const &source)
{
  expect(source.nrows() == dest.nrows());
  expect(source.ncols() == dest.ncols());
#ifdef ASGARD_USE_CUDA
  auto const success =
      cudaMemcpy2D(dest.data(), dest.stride() * sizeof(P), source.data(),
                   source.stride() * sizeof(P), source.nrows() * sizeof(P),
                   source.ncols(), cudaMemcpyDeviceToHost);
  expect(success == 0);
#else
  std::copy(source.begin(), source.end(), dest.begin());
#endif
}

} // namespace fk

//
// This would otherwise be the start of the tensors.cpp, if we were still doing
// the explicit instantiations
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

//-----------------------------------------------------------------------------
//
// fk::vector class implementation starts here
//
//-----------------------------------------------------------------------------
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector()
    : data_{nullptr}, size_{0}, ref_count_{std::make_shared<int>(0)}
{}
// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(int const size)
    : size_{size}, ref_count_{std::make_shared<int>(0)}
{
  expect(size >= 0);

  if constexpr (resrc == resource::host)
  {
    data_ = new P[size_]();
  }
  else
  {
    allocate_device(data_, size_);
  }
}

// can also do this with variadic template constructor for constness
// https://stackoverflow.com/a/5549918
// but possibly this is "too clever" for our needs right now
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(std::initializer_list<P> list)
    : size_{static_cast<int>(list.size())}, ref_count_{std::make_shared<int>(0)}
{
  if constexpr (resrc == resource::host)
  {
    data_ = new P[size_]();
    std::copy(list.begin(), list.end(), data_);
  }
  else
  {
    allocate_device(data_, size_);
    copy_to_device(data_, list.begin(), size_);
  }
}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::vector<P, mem, resrc>::vector(std::vector<P> const &v)
    : data_{new P[v.size()]}, size_{static_cast<int>(v.size())},
      ref_count_{std::make_shared<int>(0)}
{
  std::copy(v.begin(), v.end(), data_);
}

//
// matrix conversion constructor linearizes the matrix, i.e. stacks the columns
// of the matrix into a single vector
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem, resrc>::vector(
    fk::matrix<P, mem_type::owner, resrc> const &mat)
    : ref_count_{std::make_shared<int>(0)}
{
  size_ = mat.size();
  if ((*this).size() == 0)
  {
    if constexpr (resrc == resource::host)
    {
      delete[] data_;
    }
    else
    {
      delete_device(data_);
    }
    data_ = nullptr;
  }
  else
  {
    if constexpr (resrc == resource::host)
    {
      data_ = new P[mat.size()]();
      int i = 0;
      for (auto const &elem : mat)
      {
        (*this)(i++) = elem;
      }
    }
    else
    {
      allocate_device(data_, size_);
      copy_on_device(data_, mat.data(), mat.size());
    }
  }
}

// vector view constructor given a start and stop index
// modifiable view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> &vec,
                                  int const start_index, int const stop_index)
    : vector(vec, start_index, stop_index, true)
{}

// const view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &vec,
                                  int const start_index, int const stop_index)
    : vector(vec, start_index, stop_index, true)
{}

// delegating constructor to extract view from owner. overload for default case
// of viewing the entire owner
// const view version
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &a)
    : vector(a, 0, std::max(0, a.size() - 1), true)
{}

// modifiable view version
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> &a)
    : vector(a, 0, std::max(0, a.size() - 1), true)
{}

// create vector view of an existing matrix
// const version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> const &source,
                                  int const column_index, int const row_start,
                                  int const row_stop)
    : vector(source, source.get_ref_count({}), column_index, row_start,
             row_stop)
{}

// modifiable view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> &source,
                                  int const column_index, int const row_start,
                                  int const row_stop)
    : vector(source, source.get_ref_count({}), column_index, row_start,
             row_stop)
{}

template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::~vector()
{
  if constexpr (mem == mem_type::owner)
  {
    expect(ref_count_.use_count() == 1);

    if constexpr (resrc == resource::host)
    {
      delete[] data_;
    }
    else
    {
      delete_device(data_);
    }
  }
}

//
// vector copy constructor for like types (like types only)
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::vector(vector<P, mem, resrc> const &a)
    : size_{a.size_}
{
  if constexpr (mem == mem_type::owner)
  {
    ref_count_ = std::make_shared<int>(0);

    if constexpr (resrc == resource::host)
    {
      data_ = new P[a.size()];
      std::memcpy(data_, a.data(), a.size() * sizeof(P));
    }
    else
    {
      allocate_device(data_, a.size());
      copy_on_device(data_, a.data(), a.size());
    }
  }
  else
  {
    data_      = a.data();
    ref_count_ = a.ref_count_;
  }
}

//
// vector copy assignment
// this can probably be optimized better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, mem, resrc> const &a)
{
  static_assert(mem != mem_type::const_view,
                "cannot copy assign into const_view!");

  if (&a == this)
    return *this;

  expect(size() == a.size());

  if constexpr (resrc == resource::host)
  {
    std::memcpy(data_, a.data(), a.size() * sizeof(P));
  }
  else
  {
    copy_on_device(data_, a.data(), a.size());
  }

  return *this;
}

//
// vector move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc>::vector(vector<P, mem, resrc> &&a)
    : data_{a.data_}, size_{a.size_}
{
  if constexpr (mem == mem_type::owner)
  {
    expect(a.ref_count_.use_count() == 1);
  }
  ref_count_ = std::make_shared<int>(0);
  ref_count_.swap(a.ref_count_);
  a.data_ = nullptr; // b/c a's destructor will be called
  a.size_ = 0;
}

//
// vector move assignment
//
template<typename P, mem_type mem, resource resrc>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, mem, resrc> &&a)
{
  static_assert(mem != mem_type::const_view,
                "cannot move assign into const_view!");

  if (&a == this)
    return *this;

  if constexpr (mem == mem_type::owner)
  {
    expect(ref_count_.use_count() == 1);
    expect(a.ref_count_.use_count() == 1);
  }

  expect(size() == a.size());
  size_      = a.size_;
  ref_count_ = std::make_shared<int>(0);
  ref_count_.swap(a.ref_count_);
  P *const temp{data_};
  data_   = a.data_;
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// converting vector constructor
//
template<typename P, mem_type mem, resource resrc>
template<typename PP, mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem, resrc>::vector(vector<PP, omem> const &a)
    : data_{new P[a.size()]}, size_{a.size()}, ref_count_{
                                                   std::make_shared<int>(0)}
{
  for (auto i = 0; i < a.size(); ++i)
  {
    (*this)(i) = static_cast<P>(a(i));
  }
}

//
// converting vector assignment overload
// this can probably be optimized better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem, resource resrc>
template<typename PP, mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::operator=(vector<PP, omem> const &a)
{
  expect(size() == a.size());

  size_ = a.size();
  for (auto i = 0; i < a.size(); ++i)
  {
    (*this)(i) = static_cast<P>(a(i));
  }

  return *this;
}

// copy construct owner from view values
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, mem_type, typename>
fk::vector<P, mem, resrc>::vector(vector<P, omem, resrc> const &a)
    : size_(a.size()), ref_count_(std::make_shared<int>(0))
{
  if constexpr (resrc == resource::host)
  {
    data_ = new P[a.size()];
    std::memcpy(data_, a.data(), a.size() * sizeof(P));
  }
  else
  {
    allocate_device(data_, a.size());
    copy_on_device(data_, a.data(), a.size());
  }
}

// assignment owner <-> view
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc> &
fk::vector<P, mem, resrc>::operator=(vector<P, omem, resrc> const &a)
{
  expect(size() == a.size());
  if constexpr (resrc == resource::host)
  {
    std::memcpy(data_, a.data(), size() * sizeof(P));
  }
  else
  {
    copy_on_device(data_, a.data(), size());
  }
  return *this;
}

// transfer functions
// host->dev, new vector
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P, mem_type::owner, resource::device>
fk::vector<P, mem, resrc>::clone_onto_device() const

{
  fk::vector<P, mem_type::owner, resource::device> a(size());
  copy_to_device(a.data(), data(), size());
  return a;
}

// host->dev copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem, resrc> &fk::vector<P, mem, resrc>::transfer_from(
    fk::vector<P, omem, resource::host> const &a)
{
  expect(a.size() == size());
  copy_to_device(data_, a.data(), a.size());
  return *this;
}

// dev -> host, new vector
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P, mem_type::owner, resource::host>
fk::vector<P, mem, resrc>::clone_onto_host() const

{
  fk::vector<P> a(size());
  copy_to_host(a.data(), data(), size());
  return a;
}

// dev -> host copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem, resrc> &fk::vector<P, mem, resrc>::transfer_from(
    vector<P, omem, resource::device> const &a)
{
  expect(a.size() == size());
  copy_to_host(data_, a.data(), a.size());
  return *this;
}

//
// copy out of std::vector
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::operator=(std::vector<P> const &v)
{
  expect(size() == static_cast<int>(v.size()));
  std::memcpy(data_, v.data(), v.size() * sizeof(P));
  return *this;
}

//
// copy into std::vector
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
std::vector<P> fk::vector<P, mem, resrc>::to_std() const
{
  return std::vector<P>(data(), data() + size());
}

// vector subscript operator
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
P &fk::vector<P, mem, resrc>::operator()(int i)
{
  expect(i < size_);
  return data_[i];
}

template<typename P, mem_type mem, resource resrc>
template<resource, typename>
P fk::vector<P, mem, resrc>::operator()(int i) const
{
  expect(i < size_);
  return data_[i];
}

// vector comparison operators - set default tolerance above
// see https://stackoverflow.com/a/253874/6595797
// FIXME do we need to be more careful with these fp comparisons?
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator==(vector<P, omem> const &other) const
{
  if constexpr (omem == mem)
    if (&other == this)
      return true;
  if (size() != other.size())
    return false;
  for (auto i = 0; i < size(); ++i)
    if constexpr (std::is_floating_point<P>::value)
    {
      if (std::abs((*this)(i)-other(i)) > TOL)
      {
        return false;
      }
    }
    else
    {
      if ((*this)(i) != other(i))
      {
        return false;
      }
    }
  return true;
}
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator!=(vector<P, omem> const &other) const
{
  return !(*this == other);
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::vector<P, mem, resrc>::operator<(vector<P, omem> const &other) const
{
  return std::lexicographical_compare(begin(), end(), other.begin(),
                                      other.end());
}

//
// vector addition operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::operator+(vector<P, omem> const &right) const
{
  expect(size() == right.size());
  vector<P> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i) + right(i);
  return ans;
}

//
// vector subtraction operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::operator-(vector<P, omem> const &right) const
{
  expect(size() == right.size());
  vector<P> ans(size());
  for (auto i = 0; i < size(); ++i)
    ans(i) = (*this)(i)-right(i);
  return ans;
}

//
// vector*vector multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
P fk::vector<P, mem, resrc>::operator*(vector<P, omem> const &right) const
{
  expect(size() == right.size());
  int n           = size();
  int one         = 1;
  vector const &X = (*this);

  return lib_dispatch::dot(&n, X.data(), &one, right.data(), &one);
}

//
// vector*matrix multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
    fk::vector<P, mem, resrc>::operator*(fk::matrix<P, omem> const &A) const
{
  // check dimension compatibility
  expect(size() == A.nrows());

  vector const &X = (*this);
  vector<P> Y(A.ncols());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = A.stride();
  int one_i = 1;

  P zero = 0.0;
  P one  = 1.0;
  lib_dispatch::gemv("t", &m, &n, &one, A.data(), &lda, X.data(), &one_i, &zero,
                     Y.data(), &one_i);
  return Y;
}

//
// vector*scalar multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P> fk::vector<P, mem, resrc>::operator*(P const x) const
{
  vector<P> a(*this);
  int one_i = 1;
  int n     = a.size();
  P alpha   = x;

  lib_dispatch::scal(&n, &alpha, a.data(), &one_i);

  return a;
}

//
// perform the matrix kronecker product by
// interpreting vector operands/return vector
// as single column matrices.
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P> fk::vector<P, mem, resrc>::single_column_kron(
    vector<P, omem> const &right) const
{
  fk::vector<P> product((*this).size() * right.size());
  for (int i = 0; i < (*this).size(); ++i)
  {
    for (int j = 0; j < right.size(); ++j)
    {
      product(i * right.size() + j) = (*this)(i)*right(j);
    }
  }
  return product;
}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::vector<P, mem> &fk::vector<P, mem, resrc>::scale(P const x)
{
  int one_i = 1;
  int n     = this->size();
  P alpha   = x;

  lib_dispatch::scal(&n, &alpha, this->data(), &one_i);

  return *this;
}

//
// utility functions
//

//
// Prints out the values of a vector
//
// @param[in]   label   a string label printed with the output
// @param[in]   b       the vector from the batch to print out
// @return      Nothing
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::vector<P, mem, resrc>::print(std::string const label) const
{
  if constexpr (mem == mem_type::owner)
    std::cout << label << "(owner, ref_count = " << ref_count_.use_count()
              << ")" << '\n';
  else if constexpr (mem == mem_type::view)
    std::cout << label << "(view)" << '\n';
  else if constexpr (mem == mem_type::const_view)
    std::cout << label << "(const view)" << '\n';
  else
    expect(false); // above cases should cover all implemented types

  if constexpr (std::is_floating_point<P>::value)
  {
    for (auto i = 0; i < size(); ++i)
      std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                << std::right << (*this)(i);
  }
  else
  {
    for (auto i = 0; i < size(); ++i)
      std::cout << std::right << (*this)(i) << " ";
  }
  std::cout << '\n';
}

//
// Dumps to file a vector that can be read straight into octave
// Same as the matrix:: version
//
// @param[in]   label   a string label printed with the output
// @param[in]   b       the vector from the batch to print out
// @return      Nothing
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::vector<P, mem, resrc>::dump_to_octave(char const *filename) const
{
  std::ofstream ofile(filename);
  auto coutbuf = std::cout.rdbuf(ofile.rdbuf());
  for (auto i = 0; i < size(); ++i)
    std::cout << std::setprecision(12) << (*this)(i) << " ";

  std::cout.rdbuf(coutbuf);
}

//
// resize the vector
// (currently supports a subset of the std::vector.resize() interface)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::vector<P, mem_type::owner, resrc> &
fk::vector<P, mem, resrc>::resize(int const new_size)
{
  expect(new_size >= 0);
  if (new_size == this->size())
    return *this;
  P *old_data{data_};

  if constexpr (resrc == resource::host)
  {
    data_ = new P[new_size]();
    if (size() > 0 && new_size > 0)
    {
      if (size() < new_size)
        std::memcpy(data_, old_data, size() * sizeof(P));
      else
        std::memcpy(data_, old_data, new_size * sizeof(P));
    }
    delete[] old_data;
  }
  else
  {
    allocate_device(data_, new_size);
    if (size() > 0 && new_size > 0)
    {
      if (size() < new_size)
        copy_on_device(data_, old_data, size());
      else
        copy_on_device(data_, old_data, new_size);
    }
    delete_device(old_data);
  }

  size_ = new_size;
  return *this;
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::concat(vector<P, omem> const &right)
{
  int const old_size = this->size();
  int const new_size = this->size() + right.size();
  P *old_data{data_};
  data_ = new P[new_size]();
  std::memcpy(data_, old_data, old_size * sizeof(P));
  std::memcpy(data(old_size), right.data(), right.size() * sizeof(P));
  size_ = new_size;
  delete[] old_data;
  return *this;
}

// set a subvector beginning at provided index
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::vector<P, mem> &
fk::vector<P, mem, resrc>::set_subvector(int const index,
                                         fk::vector<P, omem> const sub_vector)
{
  expect(index >= 0);
  expect((index + sub_vector.size()) <= this->size());
  std::memcpy(&(*this)(index), sub_vector.data(),
              sub_vector.size() * sizeof(P));
  return *this;
}

// extract subvector, indices inclusive
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::vector<P>
fk::vector<P, mem, resrc>::extract(int const start, int const stop) const
{
  expect(start >= 0);
  expect(stop < this->size());
  expect(stop >= start);

  int const sub_size = stop - start + 1;
  fk::vector<P> sub_vector(sub_size);
  for (int i = 0; i < sub_size; ++i)
  {
    sub_vector(i) = (*this)(i + start);
  }
  return sub_vector;
}

// get number of outstanding views for an owner
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
int fk::vector<P, mem, resrc>::get_num_views() const
{
  return ref_count_.use_count() - 1;
}

// const/nonconst view constructors delegate to this private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::vector<P, mem, resrc>::vector(fk::vector<P, omem, resrc> const &vec,
                                  int const start_index, int const stop_index,
                                  bool const delegated)
    : ref_count_{vec.ref_count_}
{
  // ignore dummy argument to avoid compiler warnings
  ignore(delegated);
  data_ = nullptr;
  size_ = 0;
  if (vec.size() > 0)
  {
    expect(start_index >= 0);
    expect(stop_index < vec.size());
    expect(stop_index >= start_index);

    data_ = vec.data_ + start_index;
    size_ = stop_index - start_index + 1;
  }
}

// public const/nonconst vector view from matrix constructors delegate to
// this private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::vector<P, mem, resrc>::vector(fk::matrix<P, omem, resrc> const &source,
                                  std::shared_ptr<int> source_ref_count,
                                  int const column_index, int const row_start,
                                  int const row_stop)
    : ref_count_(source_ref_count)
{
  expect(column_index >= 0);
  expect(column_index < source.ncols());
  expect(row_start >= 0);
  expect(row_start <= row_stop);
  expect(row_stop < source.nrows());

  data_ = nullptr;
  size_ = row_stop - row_start + 1;

  if (size_ > 0)
  {
    data_ = source.data(column_index * source.stride() + row_start);
  }
}

//-----------------------------------------------------------------------------
//
// fk::matrix class implementation starts here
//
//-----------------------------------------------------------------------------

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::matrix<P, mem, resrc>::matrix()
    : data_{nullptr}, nrows_{0}, ncols_{0}, stride_{nrows_},
      ref_count_{std::make_shared<int>(0)}

{}

// right now, initializing with zero for e.g. passing in answer vectors to blas
// but this is probably slower if needing to declare in a perf. critical region
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(int const m, int const n)
    : nrows_{m}, ncols_{n}, stride_{nrows_}, ref_count_{
                                                 std::make_shared<int>(0)}

{
  expect(m >= 0);
  expect(n >= 0);

  if constexpr (resrc == resource::host)
  {
    data_ = new P[nrows() * ncols()]();
  }
  else
  {
    allocate_device(data_, nrows() * ncols());
  }
}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(
    std::initializer_list<std::initializer_list<P>> llist)
    : nrows_{static_cast<int>(llist.size())}, ncols_{static_cast<int>(
                                                  llist.begin()->size())},
      stride_{nrows_}, ref_count_{std::make_shared<int>(0)}
{
  if constexpr (resrc == resource::host)
  {
    data_       = new P[llist.size() * llist.begin()->size()]();
    int row_idx = 0;
    for (auto const &row_list : llist)
    {
      // much simpler for row-major storage
      // std::copy(row_list.begin(), row_list.end(), data(row_idx));
      int col_idx = 0;
      for (auto const &col_elem : row_list)
      {
        (*this)(row_idx, col_idx) = col_elem;
        ++col_idx;
      }
      ++row_idx;
    }
  }
  else
  {
    fk::matrix<P, mem, resource::host> const wrap(llist);
    allocate_device(data_, llist.size() * llist.begin()->size());
    copy_matrix_to_device(*this, wrap);
  }
}

// create view from owner - const view version
// delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> const &owner,
                                  int const start_row, int const stop_row,
                                  int const start_col, int const stop_col)
    : matrix(owner, start_row, stop_row, start_col, stop_col, true)
{}

// create view from owner - modifiable view version
// delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> &owner,
                                  int const start_row, int const stop_row,
                                  int const start_col, int const stop_col)
    : matrix(owner, start_row, stop_row, start_col, stop_col, true)
{}

// overload for default case - whole matrix
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> const &owner)
    : matrix(owner, 0, std::max(0, owner.nrows() - 1), 0,
             std::max(0, owner.ncols() - 1), true)
{}

template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> &owner)
    : matrix(owner, 0, std::max(0, owner.nrows() - 1), 0,
             std::max(0, owner.ncols() - 1), true)
{}

// create matrix view of an existing vector
// const version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem, resrc>::matrix(fk::vector<P, omem, resrc> const &source,
                                  int const num_rows, int const num_cols,
                                  int const start_index)
    : matrix(source, source.get_ref_count({}), num_rows, num_cols, start_index)
{}

// create matrix view of existing vector
// modifiable view version - delegates to private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(fk::vector<P, omem, resrc> &source,
                                  int const num_rows, int const num_cols,
                                  int const start_index)
    : matrix(source, source.get_ref_count({}), num_rows, num_cols, start_index)
{}

// destructor
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc>::~matrix()
{
  if constexpr (mem == mem_type::owner)
  {
    expect(ref_count_.use_count() == 1);
    if constexpr (resrc == resource::host)
    {
      delete[] data_;
    }
    else
    {
      delete_device(data_);
    }
  }
}

//
// matrix copy constructor
//
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc>::matrix(matrix<P, mem, resrc> const &a)
    : nrows_{a.nrows()}, ncols_{a.ncols()}, stride_{a.stride()}

{
  if constexpr (mem == mem_type::owner)
  {
    ref_count_ = std::make_shared<int>(0);

    if constexpr (resrc == resource::host)
    {
      data_ = new P[a.size()]();
      std::copy(a.begin(), a.end(), begin());
    }
    else
    {
      allocate_device(data_, a.size());
      copy_matrix_on_device(*this, a);
    }
  }
  else
  {
    data_      = a.data();
    ref_count_ = a.ref_count_;
  }
}

//
// matrix copy assignment
// this can probably be done better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc> &
fk::matrix<P, mem, resrc>::operator=(matrix<P, mem, resrc> const &a)
{
  static_assert(mem != mem_type::const_view,
                "cannot copy assign into const_view!");

  if (&a == this)
    return *this;

  expect((nrows() == a.nrows()) && (ncols() == a.ncols()));

  if constexpr (mem == mem_type::owner)
  {
    if constexpr (resrc == resource::host)
    {
      std::copy(a.begin(), a.end(), begin());
    }
    else
    {
      copy_matrix_on_device(*this, a);
    }
  }
  else
  {
    data_      = a.data();
    ref_count_ = a.ref_count_;
  }

  return *this;
}

// copy construct owner from view values
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(matrix<P, omem, resrc> const &a)
    : nrows_{a.nrows()}, ncols_{a.ncols()}, stride_{a.nrows()},
      ref_count_{std::make_shared<int>(0)}
{
  if constexpr (resrc == resource::host)
  {
    data_ = new P[size()]();
    std::copy(a.begin(), a.end(), begin());
  }
  else
  {
    allocate_device(data_, size());
    copy_matrix_on_device(*this, a);
  }
}

// assignment owner <-> view
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc> &
fk::matrix<P, mem, resrc>::operator=(matrix<P, omem, resrc> const &a)
{
  expect(nrows() == a.nrows());
  expect(ncols() == a.ncols());
  if constexpr (resrc == resource::host)
  {
    std::copy(a.begin(), a.end(), begin());
  }
  else
  {
    copy_matrix_on_device(*this, a);
  }
  return *this;
}

//
// converting matrix copy constructor
//
template<typename P, mem_type mem, resource resrc>
template<typename PP, mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem, resrc>::matrix(matrix<PP, omem> const &a)
    : data_{new P[a.size()]()}, nrows_{a.nrows()}, ncols_{a.ncols()},
      stride_{a.nrows()}, ref_count_{std::make_shared<int>(0)}

{
  for (auto j = 0; j < a.ncols(); ++j)
    for (auto i = 0; i < a.nrows(); ++i)
    {
      (*this)(i, j) = static_cast<P>(a(i, j));
    }
}

//
// converting matrix copy assignment
// this can probably be done better. see:
// http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
//
template<typename P, mem_type mem, resource resrc>
template<typename PP, mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::operator=(matrix<PP, omem> const &a)
{
  expect((nrows() == a.nrows()) && (ncols() == a.ncols()));

  nrows_ = a.nrows();
  ncols_ = a.ncols();
  for (auto j = 0; j < a.ncols(); ++j)
    for (auto i = 0; i < a.nrows(); ++i)
    {
      (*this)(i, j) = static_cast<P>(a(i, j));
    }
  return *this;
}

// transfer functions
// host->dev, new matrix
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::matrix<P, mem_type::owner, resource::device>
fk::matrix<P, mem, resrc>::clone_onto_device() const

{
  fk::matrix<P, mem_type::owner, resource::device> a(nrows(), ncols());
  copy_matrix_to_device(a, *this);
  return a;
}

// host->dev copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P, mem, resrc> &fk::matrix<P, mem, resrc>::transfer_from(
    fk::matrix<P, omem, resource::host> const &a)
{
  expect(a.nrows() == nrows());
  expect(a.ncols() == ncols());

  copy_matrix_to_device(*this, a);

  return *this;
}

// dev->host, new matrix
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::matrix<P, mem_type::owner, resource::host>
fk::matrix<P, mem, resrc>::clone_onto_host() const

{
  fk::matrix<P> a(nrows(), ncols());
  copy_matrix_to_host(a, *this);
  return a;
}

// dev->host copy
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P, mem, resrc> &fk::matrix<P, mem, resrc>::transfer_from(
    matrix<P, omem, resource::device> const &a)
{
  expect(a.nrows() == nrows());
  expect(a.ncols() == ncols());
  copy_matrix_to_host(*this, a);
  return *this;
}

//
// matrix move constructor
// this can probably be done better. see:
// http://stackoverflow.com/questions/3106110/what-are-move-semantics
//
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc>::matrix(matrix<P, mem, resrc> &&a)
    : data_{a.data()}, nrows_{a.nrows()}, ncols_{a.ncols()}, stride_{a.stride()}
{
  if constexpr (mem == mem_type::owner)
  {
    expect(a.ref_count_.use_count() == 1);
  }

  ref_count_ = std::make_shared<int>(0);
  ref_count_.swap(a.ref_count_);

  a.data_  = nullptr; // b/c a's destructor will be called
  a.nrows_ = 0;
  a.ncols_ = 0;
}

//
// matrix move assignment
//
template<typename P, mem_type mem, resource resrc>
fk::matrix<P, mem, resrc> &
fk::matrix<P, mem, resrc>::operator=(matrix<P, mem, resrc> &&a)
{
  static_assert(mem != mem_type::const_view,
                "cannot move assign into const_view!");

  if (&a == this)
    return *this;

  expect((nrows() == a.nrows()) &&
         (ncols() == a.ncols() && stride() == a.stride()));

  // check for destination orphaning; see below
  if constexpr (mem == mem_type::owner)
  {
    expect(ref_count_.use_count() == 1 && a.ref_count_.use_count() == 1);
  }
  ref_count_ = std::make_shared<int>(0);
  ref_count_.swap(a.ref_count_);

  P *temp{data_};
  // this would orphan views on the destination
  data_   = a.data();
  a.data_ = temp; // b/c a's destructor will be called
  return *this;
}

//
// copy out of fk::vector - assumes the vector is column-major
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::operator=(fk::vector<P, omem> const &v)
{
  expect(nrows() * ncols() == v.size());

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      (*this)(i, j) = v(j + i * ncols());

  return *this;
}

//
// matrix subscript operator - (row, col)
// see c++faq:
// https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
P &fk::matrix<P, mem, resrc>::operator()(int const i, int const j)
{
  expect(i < nrows() && j < ncols());
  return *(data(i, j));
}

template<typename P, mem_type mem, resource resrc>
template<resource, typename>
P fk::matrix<P, mem, resrc>::operator()(int const i, int const j) const
{
  expect(i < nrows() && j < ncols());
  return *(data(i, j));
}

// matrix comparison operators - set default tolerance above
// see https://stackoverflow.com/a/253874/6595797
// FIXME we may need to be more careful with these comparisons
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::matrix<P, mem, resrc>::operator==(matrix<P, omem> const &other) const
{
  if constexpr (mem == omem)
  {
    if (&other == this)
      return true;
  }

  if (nrows() != other.nrows() || ncols() != other.ncols())
    return false;
  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      if constexpr (std::is_floating_point<P>::value)
      {
        if (std::abs((*this)(i, j) - other(i, j)) > TOL)
        {
          return false;
        }
      }
      else
      {
        if ((*this)(i, j) != other(i, j))
        {
          return false;
        }
      }
  return true;
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::matrix<P, mem, resrc>::operator!=(matrix<P, omem> const &other) const
{
  return !(*this == other);
}

template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
bool fk::matrix<P, mem, resrc>::operator<(matrix<P, omem> const &other) const
{
  return std::lexicographical_compare(this->begin(), this->end(), other.begin(),
                                      other.end());
}

//
// matrix addition operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P>
fk::matrix<P, mem, resrc>::operator+(matrix<P, omem> const &right) const
{
  expect(nrows() == right.nrows() && ncols() == right.ncols());

  matrix<P> ans(nrows(), ncols());
  ans.nrows_ = nrows();
  ans.ncols_ = ncols();

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      ans(i, j) = (*this)(i, j) + right(i, j);

  return ans;
}

//
// matrix subtraction operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P>
fk::matrix<P, mem, resrc>::operator-(matrix<P, omem> const &right) const
{
  expect(nrows() == right.nrows() && ncols() == right.ncols());

  matrix<P> ans(nrows(), ncols());
  ans.nrows_ = nrows();
  ans.ncols_ = ncols();

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      ans(i, j) = (*this)(i, j) - right(i, j);

  return ans;
}

//
// matrix*scalar multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::matrix<P> fk::matrix<P, mem, resrc>::operator*(P const right) const
{
  matrix<P> ans(nrows(), ncols());
  ans.nrows_ = nrows();
  ans.ncols_ = ncols();

  for (auto j = 0; j < ncols(); ++j)
    for (auto i = 0; i < nrows(); ++i)
      ans(i, j) = (*this)(i, j) * right;

  return ans;
}

//
// matrix*vector multiplication operator
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::vector<P>
    fk::matrix<P, mem, resrc>::operator*(fk::vector<P, omem> const &right) const
{
  // check dimension compatibility
  expect(ncols() == right.size());

  matrix<P, mem> const &A = (*this);
  vector<P> Y(A.nrows());

  int m     = A.nrows();
  int n     = A.ncols();
  int lda   = A.stride();
  int one_i = 1;

  P one  = 1.0;
  P zero = 0.0;
  lib_dispatch::gemv("n", &m, &n, &one, A.data(), &lda, right.data(), &one_i,
                     &zero, Y.data(), &one_i);

  return Y;
}

//
// matrix*matrix multiplication operator C[m,n] = A[m,k] * B[k,n]
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P>
    fk::matrix<P, mem, resrc>::operator*(matrix<P, omem> const &B) const
{
  expect(ncols() == B.nrows()); // k == k

  // just aliases for easier reading
  matrix const &A = (*this);
  int m           = A.nrows();
  int n           = B.ncols();
  int k           = B.nrows();

  matrix<P> C(m, n);

  int lda = A.stride();
  int ldb = B.stride();
  int ldc = C.stride();

  P one  = 1.0;
  P zero = 0.0;
  lib_dispatch::gemm("n", "n", &m, &n, &k, &one, A.data(), &lda, B.data(), &ldb,
                     &zero, C.data(), &ldc);

  return C;
}

/* in-place matrix transpose for column major data layout */
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::matrix<P, mem, resrc> &fk::matrix<P, mem, resrc>::transpose()
{
  /* empty matrix */
  if (size() == 0)
    return *this;

  /* vector pretending to be a matrix */
  if (nrows_ == 1 || ncols_ == 1)
  {
    std::swap(nrows_, ncols_);
    stride_ = nrows_;
    return *this;
  }

  /* square matrix */
  if (nrows_ == ncols_)
  {
    for (int r = 0; r < nrows_; ++r)
    {
      for (int c = 0; c < r; ++c)
      {
        std::swap(data_[c * nrows_ + r], data_[r * nrows_ + c]);
      }
    }
    return *this;
  }

  /* spot for each element, true ~ visited, false ~ unvisited */
  std::vector<bool> visited(size() - 2, false);

  /* Given index "pos" in a linear array interpreted as a matrix of "nrows_"
     rows, n_cols_ columns, and column-major data layout, return the linear
     index position of the element in the matrix's transpose */
  auto const remap_index = [this](int const pos) -> int {
    int const row         = pos % nrows_;
    int const col         = pos / nrows_;
    int const destination = row * ncols_ + col;
    return destination;
  };

  /* The first and last elements never change position and can be ignored */
  for (int pos = 1; pos < size() - 1; ++pos)
  {
    if (visited[pos])
      continue;

    P save = data_[pos];

    int next_pos = remap_index(pos);

    while (!visited[next_pos])
    {
      std::swap(save, data_[next_pos]);
      visited[next_pos] = true;
      next_pos          = remap_index(next_pos);
    }
  }

  std::swap(nrows_, ncols_);
  stride_ = nrows_;

  return *this;
}

// Simple quad-loop kron prod
// @return the product
//
// FIXME this is NOT optimized.
// we will use the other methods
// for performance-critical (large)
// krons
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, resource, typename>
fk::matrix<P> fk::matrix<P, mem, resrc>::kron(matrix<P, omem> const &B) const
{
  fk::matrix<P> C(nrows() * B.nrows(), ncols() * B.ncols());
  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
    {
      for (auto k = 0; k < B.nrows(); ++k)
      {
        for (auto l = 0; l < B.ncols(); ++l)
        {
          C((i * B.nrows() + k), (j * B.ncols() + l)) +=
              (*this)(i, j) * B(k, l);
        }
      }
    }
  }
  return C;
}

//
// Invert a square matrix (overwrites original)
// disabled for non-fp types; haven't written a routine to do it
// @return  the inverted matrix
//
template<typename P, mem_type mem, resource resrc>
template<typename U, typename, mem_type, typename, resource, typename>
fk::matrix<P, mem> &fk::matrix<P, mem, resrc>::invert()
{
  expect(nrows() == ncols());

  int *ipiv{new int[ncols()]};
  int lwork{nrows() * ncols()};
  int lda = stride();
  P *work{new P[nrows() * ncols()]};
  int info;

  lib_dispatch::getrf(&ncols_, &ncols_, data(0, 0), &lda, ipiv, &info);
  lib_dispatch::getri(&ncols_, data(0, 0), &lda, ipiv, work, &lwork, &info);

  delete[] ipiv;
  delete[] work;
  return *this;
}

//
// Get the determinant of the matrix  (non destructive)
// (based on src/Numerics/DeterminantOperators.h)
// (note possible problems with over/underflow
// - see Ed's emails 12/5/16, 10/14/16, 10/10/16.
// how is this handled / is it necessary in production?
// possibly okay for small KxK matrices - can build in a check/warning)
//
//
// disabled for non-float types; haven't written a routine to do it
//
// @param[in]   mat   integer matrix (walker) to get determinant from
// @return  the determinant (type double)
//
template<typename P, mem_type mem, resource resrc>
template<typename U, typename, resource, typename>
P fk::matrix<P, mem, resrc>::determinant() const
{
  expect(nrows() == ncols());

  matrix<P, mem_type::owner> temp(*this); // get temp copy to do LU
  int *ipiv{new int[ncols()]};
  int info;
  int n   = temp.ncols();
  int lda = temp.stride();

  lib_dispatch::getrf(&n, &n, temp.data(0, 0), &lda, ipiv, &info);

  P det    = 1.0;
  int sign = 1;
  for (auto i = 0; i < nrows(); ++i)
  {
    if (ipiv[i] != i + 1)
      sign *= -1;
    det *= temp(i, i);
  }
  det *= static_cast<P>(sign);
  delete[] ipiv;
  return det;
}

//
// Update a specific col of a matrix, given a fk::vector<P> (overwrites
// original)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::update_col(int const col_idx,
                                      fk::vector<P, omem> const &v)
{
  expect(nrows() == static_cast<int>(v.size()));
  expect(col_idx < ncols());

  int n{v.size()};
  int one{1};
  int stride = 1;

  lib_dispatch::copy(&n, v.data(), &one, data(0, col_idx), &stride);

  return *this;
}

//
// Update a specific col of a matrix, given a std::vector (overwrites original)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::update_col(int const col_idx,
                                      std::vector<P> const &v)
{
  expect(nrows() == static_cast<int>(v.size()));
  expect(col_idx < ncols());

  int n{static_cast<int>(v.size())};
  int one{1};
  int stride = 1;

  lib_dispatch::copy(&n, const_cast<P *>(v.data()), &one, data(0, col_idx),
                     &stride);

  return *this;
}

//
// Update a specific row of a matrix, given a fk::vector<P> (overwrites
// original)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::update_row(int const row_idx,
                                      fk::vector<P, omem> const &v)
{
  expect(ncols() == v.size());
  expect(row_idx < nrows());

  int n{v.size()};
  int one{1};
  int lda = stride();

  lib_dispatch::copy(&n, v.data(), &one, data(row_idx, 0), &lda);

  return *this;
}

//
// Update a specific row of a matrix, given a std::vector (overwrites original)
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::update_row(int const row_idx,
                                      std::vector<P> const &v)
{
  expect(ncols() == static_cast<int>(v.size()));
  expect(row_idx < nrows());

  int n{static_cast<int>(v.size())};
  int one{1};
  int lda = stride();

  lib_dispatch::copy(&n, const_cast<P *>(v.data()), &one, data(row_idx, 0),
                     &lda);

  return *this;
}

//
// Resize, clearing all data
//
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
fk::matrix<P, mem_type::owner, resrc> &
fk::matrix<P, mem, resrc>::clear_and_resize(int const rows, int const cols)
{
  expect(ref_count_.use_count() == 1);

  expect(rows >= 0);
  expect(cols >= 0);
  if (rows == 0 || cols == 0)
    expect(cols == rows);

  if constexpr (resrc == resource::host)
  {
    delete[] data_;
    data_ = new P[rows * cols]();
  }
  else
  {
    delete_device(data_);
    allocate_device(data_, rows * cols);
  }

  nrows_  = rows;
  ncols_  = cols;
  stride_ = nrows_;
  return *this;
}

//
// Set a submatrix within the matrix, given another (smaller) matrix
//
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename, resource, typename>
fk::matrix<P, mem> &
fk::matrix<P, mem, resrc>::set_submatrix(int const row_idx, int const col_idx,
                                         matrix<P, omem> const &submatrix)
{
  expect(row_idx >= 0);
  expect(col_idx >= 0);
  expect(row_idx + submatrix.nrows() <= nrows());
  expect(col_idx + submatrix.ncols() <= ncols());

  matrix &matrix = *this;
  for (auto i = 0; i < submatrix.nrows(); ++i)
  {
    for (auto j = 0; j < submatrix.ncols(); ++j)
    {
      matrix(i + row_idx, j + col_idx) = submatrix(i, j);
    }
  }
  return matrix;
}

//
// Extract a rectangular submatrix from within the matrix
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
fk::matrix<P>
fk::matrix<P, mem, resrc>::extract_submatrix(int const row_idx,
                                             int const col_idx,
                                             int const num_rows,
                                             int const num_cols) const
{
  expect(row_idx >= 0);
  expect(col_idx >= 0);
  expect(row_idx + num_rows <= nrows());
  expect(col_idx + num_cols <= ncols());

  matrix<P> submatrix(num_rows, num_cols);
  auto matrix = *this;
  for (auto i = 0; i < num_rows; ++i)
  {
    for (auto j = 0; j < num_cols; ++j)
    {
      submatrix(i, j) = matrix(i + row_idx, j + col_idx);
    }
  }

  return submatrix;
}

// Prints out the values of a matrix
// @return  Nothing
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::matrix<P, mem, resrc>::print(std::string label) const
{
  if constexpr (mem == mem_type::owner)
    std::cout << label << "(owner, "
              << "outstanding views == " << std::to_string(get_num_views())
              << ")" << '\n';

  else if constexpr (mem == mem_type::view)
    std::cout << label << "(view, "
              << "stride == " << std::to_string(stride()) << ")" << '\n';

  else if constexpr (mem == mem_type::const_view)
    std::cout << label << "(const view, "
              << "stride == " << std::to_string(stride()) << ")" << '\n';
  else
    expect(false); // above cases cover all implemented mem types

  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
    {
      if constexpr (std::is_floating_point<P>::value)
      {
        std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                  << std::right << (*this)(i, j);
      }
      else
      {
        std::cout << (*this)(i, j) << " ";
      }
    }
    std::cout << '\n';
  }
}

//
// Dumps to file a matrix that can be read data straight into octave
// e.g.
//
//      dump_to_matrix ("A.dat");
//      ...
//      octave> load A.dat
//
// @return  Nothing
//
template<typename P, mem_type mem, resource resrc>
template<resource, typename>
void fk::matrix<P, mem, resrc>::dump_to_octave(char const *filename) const
{
  std::ofstream ofile(filename);
  auto coutbuf = std::cout.rdbuf(ofile.rdbuf());
  for (auto i = 0; i < nrows(); ++i)
  {
    for (auto j = 0; j < ncols(); ++j)
      std::cout << std::setprecision(12) << (*this)(i, j) << " ";

    std::cout << std::setprecision(4) << '\n';
  }
  std::cout.rdbuf(coutbuf);
}

// get number of outstanding views for an owner
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename>
int fk::matrix<P, mem, resrc>::get_num_views() const
{
  return ref_count_.use_count() - 1;
}

// public const/nonconst view constructors delegate to this private
// constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type, typename, mem_type omem>
fk::matrix<P, mem, resrc>::matrix(fk::matrix<P, omem, resrc> const &owner,
                                  int const start_row, int const stop_row,
                                  int const start_col, int const stop_col,
                                  bool const delegated)
    : ref_count_(owner.ref_count_)
{
  ignore(delegated);
  data_   = nullptr;
  nrows_  = 0;
  ncols_  = 0;
  stride_ = 0;

  int const view_rows = stop_row - start_row + 1;
  int const view_cols = stop_col - start_col + 1;
  if (owner.size() > 0)
  {
    expect(start_row >= 0);
    expect(start_col >= 0);
    expect(stop_col < owner.ncols());
    expect(stop_row < owner.nrows());
    expect(stop_row >= start_row);

    data_   = owner.data(start_row, start_col);
    nrows_  = view_rows;
    ncols_  = view_cols;
    stride_ = owner.stride();
  }
}

// public const/nonconst matrix view from vector constructors delegate to
// this private constructor
template<typename P, mem_type mem, resource resrc>
template<mem_type omem, mem_type, typename>
fk::matrix<P, mem, resrc>::matrix(fk::vector<P, omem, resrc> const &source,
                                  std::shared_ptr<int> source_ref_count,
                                  int const num_rows, int const num_cols,
                                  int const start_index)
    : ref_count_(source_ref_count)
{
  expect(start_index >= 0);
  expect(num_rows > 0);
  expect(num_cols > 0);

  int const size = num_rows * num_cols;
  expect(start_index + size <= source.size());

  data_   = nullptr;
  nrows_  = 0;
  ncols_  = 0;
  stride_ = 0;

  if (size > 0)
  {
    data_   = source.data(start_index);
    nrows_  = num_rows;
    ncols_  = num_cols;
    stride_ = num_rows;
  }
}

template<typename P, mem_type mem, resource resrc>
template<typename T, typename R>
class fk::matrix<P, mem, resrc>::matrix_iterator
{
public:
  using self_type         = matrix_iterator;
  using value_type        = P;
  using reference         = R;
  using pointer           = T;
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = int;
  matrix_iterator(pointer ptr, int const stride, int const rows)
      : ptr_(ptr), start_(ptr), stride_(stride), rows_(rows)
  {}

  difference_type increment()
  {
    difference_type const next_pos = ptr_ - start_ + 1;

    if (next_pos % rows_ != 0)
    {
      return 1;
    }
    else
    {
      start_ += stride_;
      return stride_ - rows_ + 1;
    }
  }

  self_type operator++(int)
  {
    self_type i = *this;
    ptr_ += increment();
    return i;
  }
  self_type operator++()
  {
    ptr_ += increment();
    return *this;
  }

  reference operator*() const { return *ptr_; }
  pointer operator->() const { return ptr_; }
  bool operator==(const self_type &rhs) const { return ptr_ == rhs.ptr_; }
  bool operator!=(const self_type &rhs) const { return ptr_ != rhs.ptr_; }

private:
  pointer ptr_;
  pointer start_;
  int stride_;
  int rows_;
};

template<typename P, mem_type left_mem, mem_type right_mem>
void debug_compare(fk::matrix<P, left_mem> const &left,
                   fk::matrix<P, right_mem> const &right)
{
  expect(left.nrows() == right.nrows());
  expect(left.ncols() == right.ncols());

  static std::string const red("\033[0;31m");
  static std::string const reset("\033[0m");

  for (auto i = 0; i < left.nrows(); ++i)
  {
    for (auto j = 0; j < left.ncols(); ++j)
    {
      if constexpr (std::is_floating_point<P>::value)
      {
        if (std::abs(left(i, j) - right(i, j)) > TOL)
        {
          std::cout << red;
        }

        std::cout << std::setw(12) << std::setprecision(4) << std::scientific
                  << std::right << left(i, j) << reset;
      }
      else
      {
        if (left(i, j) != right(i, j))
        {
          std::cout << red;
        }
        std::cout << std::right << left(i, j) << reset << " ";
      }
    }

    std::cout << '\n';
  }
}
