#pragma once
#include "build_info.hpp"
#include "elements.hpp"

#ifdef ASGARD_USE_MPI
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include "mpi.h"
#pragma GCC diagnostic pop
#endif

#include <list>
#include <map>
#include <vector>

// simple struct for representing a range within the element grid
struct grid_limits
{
  grid_limits(int const start, int const stop) : start(start), stop(stop){};
  grid_limits(grid_limits const &l) : start(l.start), stop(l.stop){};
  grid_limits(grid_limits const &&l) : start(l.start), stop(l.stop){};

  int size() const { return stop - start + 1; }
  bool operator==(grid_limits const &rhs) const
  {
    return start == rhs.start && stop == rhs.stop;
  }
  bool operator<(grid_limits const &rhs) const
  {
    return std::tie(start, stop) < std::tie(rhs.start, rhs.stop);
  }
  int const start;
  int const stop;
};

// this struct is designed to store information about a rank's assigned portion
// of the element grid.
//
// start and stop members are inclusive global indices of the element grid.
//
// translation functions are provided for mapping global<->local indices.

using index_mapper = std::function<int(int const)>;
class element_subgrid
{
public:
  element_subgrid(int const row_start, int const row_stop, int const col_start,
                  int const col_stop)
      : row_start(row_start), row_stop(row_stop), col_start(col_start),
        col_stop(col_stop)
  {
    expect(row_start >= 0);
    expect(row_stop >= row_start);
    expect(col_start >= 0);
    expect(col_stop >= col_start);
  };
  element_subgrid(element_subgrid const &e)
      : row_start(e.row_start), row_stop(e.row_stop), col_start(e.col_start),
        col_stop(e.col_stop){};
  element_subgrid(element_subgrid const &&e)
      : row_start(e.row_start), row_stop(e.row_stop), col_start(e.col_start),
        col_stop(e.col_stop){};
  bool operator==(const element_subgrid &rhs) const
  {
    return row_start == rhs.row_start && row_stop == rhs.row_stop &&
           col_start == rhs.col_start && col_stop == rhs.col_stop;
  }

  int64_t size() const { return static_cast<int64_t>(nrows()) * ncols(); };

  int nrows() const { return row_stop - row_start + 1; }
  int ncols() const { return col_stop - col_start + 1; }

  // translation from local/global x and y
  int to_global_row(int const local_row) const
  {
    expect(local_row >= 0);
    expect(local_row < nrows());
    return local_row + row_start;
  }
  int to_global_col(int const local_col) const
  {
    expect(local_col >= 0);
    expect(local_col < ncols());
    return local_col + col_start;
  }
  int to_local_row(int const global_row) const
  {
    expect(global_row >= 0);
    int const local = global_row - row_start;
    expect(local >= 0);
    expect(local < nrows());
    return local;
  };
  int to_local_col(int const global_col) const
  {
    expect(global_col >= 0);
    int const local = global_col - col_start;
    expect(local >= 0);
    expect(local < ncols());
    return local;
  };

  // shims for when we need to pass the above to functions
  index_mapper get_local_col_map() const
  {
    return
        [this](int const global_index) { return to_local_col(global_index); };
  }
  index_mapper get_local_row_map() const
  {
    return
        [this](int const global_index) { return to_local_row(global_index); };
  }

  int const row_start;
  int const row_stop;
  int const col_start;
  int const col_stop;
};

// -- funcs for distributing solution vector

// helper for determining the number of subgrid columns given
// a number of ranks.
//
// determine the side lengths that will give us the "squarest" rectangles
// possible
inline int get_num_subgrid_cols(int const num_ranks)
{
  expect(num_ranks > 0);
  int trial_factor = static_cast<int>(std::floor(std::sqrt(num_ranks)));
  while (trial_factor > 0)
  {
    int const other_factor = num_ranks / trial_factor;
    if (trial_factor * other_factor == num_ranks)
    {
      return std::max(trial_factor, other_factor);
    }
    trial_factor++;
  }
  // I believe this is mathematically impossible...
  expect(false);
  return 0;
}

// should be invoked once on startup to
// initialize distribution libraries
//
// return: [my_rank, num_ranks]
std::array<int, 2> initialize_distribution();

// should be invoked once on exit
void finalize_distribution();

// get node-local rank
int get_local_rank();
// get overall rank
int get_rank();
// get number of ranks overall
int get_num_ranks();

// this struct will use node-local ranks to ensure only
// one rank prints some input val:
//
// usage: node_out() << "something you want to appear once per node"
//
struct node_out
{
  template<typename T>
  node_out &operator<<(T val)
  {
#ifdef ASGARD_USE_MPI
    if (!get_local_rank())
      std::cout << val;
    return *this;
#endif
    std::cout << val;
    return *this;
  }
};

// given a rank, determine element subgrid assigned to that rank
element_subgrid get_subgrid(int const num_ranks, int const my_rank,
                            elements::table const &table);

// map ranks to assigned subgrids
// code assumes no subgrid will be larger than rank 0's.
using distribution_plan = std::map<int, element_subgrid>;
distribution_plan get_plan(int const num_ranks, elements::table const &table);

enum class message_direction
{
  send,
  receive
};

// represent a point-to-point message within or across distribution plans.
// target is the sender rank for a receive, and receive rank for send
// old range describes the global indices (inclusive) that will be transmitted
// new range describes the global indices (inclusive) for receiving
struct message
{
  message(message_direction const message_dir, int const target,
          grid_limits const &source_range, grid_limits const &dest_range)
      : message_dir(message_dir), target(target), source_range(source_range),
        dest_range(dest_range)
  {
    assert(source_range.size() == dest_range.size());
  }
  // within the same distro plan, only need one range
  // global indices are consistent
  message(message_direction const message_dir, int const target,
          grid_limits const source_range)
      : message_dir(message_dir), target(target), source_range(source_range),
        dest_range(source_range)
  {}

  message(message const &other) = default;
  message(message &&other)      = default;

  bool operator==(message const &oth) const
  {
    return (message_dir == oth.message_dir && target == oth.target &&
            source_range == oth.source_range && dest_range == oth.dest_range);
  }

  message_direction const message_dir;
  int const target;
  grid_limits const source_range;
  grid_limits const dest_range;
};

// reduce the results of a subgrid row
template<typename P>
void reduce_results(fk::vector<P> const &source, fk::vector<P> &dest,
                    distribution_plan const &plan, int const my_rank);

// generate a message list for each rank for exchange_results function;
// conceptually an internal component function, exposed for testing
std::vector<std::vector<message>> const
generate_messages(distribution_plan const &plan);

// exchange results between subgrid rows
template<typename P>
void exchange_results(fk::vector<P> const &source, fk::vector<P> &dest,
                      int const segment_size, distribution_plan const &plan,
                      int const my_rank);

// gather errors from all local ranks for printing
template<typename P>
std::array<fk::vector<P>, 2>
gather_errors(P const root_mean_squared, P const relative);

// gather final answer at end of run from all ranks
template<typename P>
std::vector<P>
gather_results(fk::vector<P> const &my_results, distribution_plan const &plan,
               int const my_rank, int const element_segment_size);

// helper func
template<typename P>
double get_MB(int64_t const num_elems)
{
  expect(num_elems > 0);
  double const bytes = num_elems * sizeof(P);
  double const MB    = bytes * 1e-6;
  return MB;
}

// -- funcs for adaptivity/redistribution

// find maximum element in plan using each rank's local max
template<typename P>
P get_global_max(P const my_max, distribution_plan const &plan);

// merge my element table additions/deletions with other nodes
std::vector<int64_t>
distribute_table_changes(std::vector<int64_t> const &my_changes,
                         distribution_plan const &plan);

// generate messages for redistribute_vector
// conceptually private, exposed for testing

// elem remap: new element index -> old grid start,stop
std::vector<std::list<message>>
generate_messages_remap(distribution_plan const &old_plan,
                        distribution_plan const &new_plan,
                        std::map<int64_t, grid_limits> const &elem_remap);

// redistribute: after adapting distribution plan, ensure all ranks have
// correct existing values for assigned subgrids

// preconditions: old plan and new plan sizes match (same number of ranks)
// also, elements must either be appended to the element grid (refinement)
// or deleted from the middle of the grid with left shift to fill (coarsening)
template<typename P>
fk::vector<P>
redistribute_vector(fk::vector<P> const &old_x,
                    distribution_plan const &old_plan,
                    distribution_plan const &new_plan,
                    std::map<int64_t, grid_limits> const &elem_remap);
