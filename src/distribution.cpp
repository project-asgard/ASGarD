#include "distribution.hpp"
#include "lib_dispatch.hpp"
#include "tools.hpp"

#include <cmath>
#include <csignal>
#include <list>
#include <numeric>

#ifdef ASGARD_USE_MPI
struct distribution_handler
{
  distribution_handler() {}

  void set_global_comm(MPI_Comm const &comm)
  {
    auto const status = MPI_Comm_dup(comm, &global_comm);
    expect(status == 0);
  }
  MPI_Comm get_global_comm() const { return global_comm; }

private:
  MPI_Comm global_comm = MPI_COMM_WORLD;
};
static distribution_handler distro_handle;
#endif

int get_local_rank()
{
#ifdef ASGARD_USE_MPI
  static auto const rank = []() {
    MPI_Comm local_comm;
    auto success = MPI_Comm_split_type(distro_handle.get_global_comm(),
                                       MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                       &local_comm);
    expect(success == 0);
    int local_rank;
    success = MPI_Comm_rank(local_comm, &local_rank);
    expect(success == 0);
    success = MPI_Comm_free(&local_comm);
    expect(success == 0);
    return local_rank;
  }();
  return rank;
#endif
  return 0;
}

int get_rank()
{
#ifdef ASGARD_USE_MPI
  static int const rank = []() {
    int my_rank;
    auto const status =
        MPI_Comm_rank(distro_handle.get_global_comm(), &my_rank);
    expect(status == 0);
    return my_rank;
  }();
  return rank;
#endif
  return 0;
}

int get_num_ranks()
{
#ifdef ASGARD_USE_MPI
  static int const num_ranks = []() {
    int num_ranks;
    auto const status =
        MPI_Comm_size(distro_handle.get_global_comm(), &num_ranks);
    expect(status == 0);
    return num_ranks;
  }();
  return num_ranks;
#endif
  return 1;
}

// to simplify distribution, we have designed the code
// to run with even and/or perfect square number of ranks.

// if run with odd and nonsquare number of ranks, the closest smaller
// even number of ranks will be used by the application. this is the
// "effective" number of ranks returned by this lambda
auto const num_effective_ranks = [](int const num_ranks) {
  if (std::sqrt(num_ranks) == std::floor(std::sqrt(num_ranks)) ||
      num_ranks % 2 == 0)
  {
    return num_ranks;
  }
  return num_ranks - 1;
};

#ifdef ASGARD_USE_MPI
static void terminate_all_ranks(int signum)
{
  MPI_Abort(distro_handle.get_global_comm(), signum);
  exit(signum);
}
#endif

std::array<int, 2> initialize_distribution()
{
  static bool init_done = false;
  expect(!init_done);
#ifdef ASGARD_USE_MPI
  signal(SIGABRT, terminate_all_ranks);
  auto status = MPI_Init(NULL, NULL);

  init_done = true;
  expect(status == 0);

  int num_ranks;
  status = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  expect(status == 0);
  int my_rank;
  status = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  expect(status == 0);

  auto const num_participating = num_effective_ranks(num_ranks);
  bool const participating     = my_rank < num_participating;
  int const comm_color         = participating ? 1 : MPI_UNDEFINED;
  MPI_Comm effective_communicator;
  auto success = MPI_Comm_split(MPI_COMM_WORLD, comm_color, my_rank,
                                &effective_communicator);
  expect(success == 0);

  status = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  expect(status == 0);

  if (effective_communicator != MPI_COMM_NULL)
  {
    distro_handle.set_global_comm(effective_communicator);
    initialize_libraries(get_local_rank());
  }

  return {my_rank, num_participating};

#endif

  return {0, 1};
}

void finalize_distribution()
{
#ifdef ASGARD_USE_MPI
  auto const status = MPI_Finalize();
  expect(status == 0);
#endif
}

// divide element grid into rectangular sub-areas, which will be assigned to
// each rank. require number of ranks to be a perfect square or an even number;
// otherwise, we will ignore (leave unused) the highest rank.
element_subgrid get_subgrid(int const num_ranks, int const my_rank,
                            elements::table const &table)
{
  expect(num_ranks > 0);

  expect(num_ranks % 2 == 0 || num_ranks == 1 ||
         std::sqrt(num_ranks) == std::floor(std::sqrt(num_ranks)));
  expect(my_rank >= 0);
  expect(my_rank < num_ranks);
  expect(table.size() >= num_ranks);

  if (num_ranks == 1)
  {
    return element_subgrid(0, table.size() - 1, 0, table.size() - 1);
  }

  int const num_subgrid_cols = get_num_subgrid_cols(num_ranks);
  int const num_subgrid_rows = num_ranks / num_subgrid_cols;

  // determine which subgrid of the element grid belongs to my rank
  int const grid_row_index = my_rank / num_subgrid_cols;
  int const grid_col_index = my_rank % num_subgrid_cols;

  // split the elements into subgrids
  int const left_over_cols = table.size() % num_subgrid_cols;
  int const subgrid_width  = table.size() / num_subgrid_cols;
  int const left_over_rows = table.size() % num_subgrid_rows;
  int const subgrid_height = table.size() / num_subgrid_rows;

  // define the bounds of my subgrid
  int const start_col =
      grid_col_index * subgrid_width + std::min(grid_col_index, left_over_cols);
  int const start_row = grid_row_index * subgrid_height +
                        std::min(grid_row_index, left_over_rows);
  int const stop_col =
      start_col + subgrid_width + (left_over_cols > grid_col_index ? 1 : 0) - 1;
  int const stop_row = start_row + subgrid_height +
                       (left_over_rows > grid_row_index ? 1 : 0) - 1;

  return element_subgrid(start_row, stop_row, start_col, stop_col);
}

// distribution plan is a mapping from rank -> assigned subgrid
distribution_plan get_plan(int const num_ranks, elements::table const &table)
{
  expect(num_ranks > 0);
  expect(table.size() > 0);
  auto const num_splits = num_effective_ranks(num_ranks);

  distribution_plan plan;
  for (int i = 0; i < num_splits; ++i)
  {
    plan.emplace(i, get_subgrid(num_splits, i, table));
  }

  return plan;
}

/* this function determines the subgrid row dependencies for each subgrid column
 *
 * the return vector is num_subgrid_columns in length. Element "x" in this
 * vector describes the subgrid rows holding data that members of subgrid column
 * "x" need to receive, as well as the global indices of that data in the
 * solution vector  */
using rows_to_range = std::map<int, grid_limits>;
static std::vector<rows_to_range>
find_column_dependencies(std::vector<int> const &row_boundaries,
                         std::vector<int> const &column_boundaries)
{
  // contains an element for each subgrid column describing
  // the subgrid rows, and associated ranges, that the column
  // members will need information from
  std::vector<rows_to_range> column_dependencies(column_boundaries.size());

  // start at the first row and column interval
  // col_start is the first index in this column interval
  int col_start = 0;
  for (int c = 0; c < static_cast<int>(column_boundaries.size()); ++c)
  {
    int row_start = 0;
    // the stop vectors represent the end of a range
    int const column_end = column_boundaries[c];
    for (int r = 0; r < static_cast<int>(row_boundaries.size()); ++r)
    {
      int const row_end = row_boundaries[r];
      // if the row interval falls within the column interval
      if ((col_start >= row_start && col_start <= row_end) ||
          (row_start >= col_start && row_start <= column_end))
      {
        // emplace the section of the row interval that falls within the column
        // interval
        column_dependencies[c].emplace(
            r, grid_limits(std::max(row_start, col_start),
                           std::min(row_end, column_end)));
      }
      // the beginning of the next interval is one more than the end of the
      // previous
      row_start = row_end + 1;
    }
    col_start = column_end + 1;
  }
  return column_dependencies;
}

template<typename P>
void reduce_results(fk::vector<P> const &source, fk::vector<P> &dest,
                    distribution_plan const &plan, int const my_rank)
{
  expect(source.size() == dest.size());
  expect(my_rank >= 0);
  expect(my_rank < static_cast<int>(plan.size()));

#ifdef ASGARD_USE_MPI
  if (plan.size() == 1)
  {
    fm::copy(source, dest);
    return;
  }

  fm::scal(static_cast<P>(0.0), dest);
  int const num_cols = get_num_subgrid_cols(plan.size());

  int const my_row = my_rank / num_cols;
  int const my_col = my_rank % num_cols;

  MPI_Comm row_communicator;
  MPI_Comm const global_communicator = distro_handle.get_global_comm();

  auto success =
      MPI_Comm_split(global_communicator, my_row, my_col, &row_communicator);
  expect(success == 0);

  MPI_Datatype const mpi_type =
      std::is_same<P, double>::value ? MPI_DOUBLE : MPI_FLOAT;
  success = MPI_Allreduce((void *)source.data(), (void *)dest.data(),
                          source.size(), mpi_type, MPI_SUM, row_communicator);
  expect(success == 0);

  success = MPI_Comm_free(&row_communicator);
  expect(success == 0);

#else
  fm::copy(source, dest);
  return;
#endif
}

//
// -- below functionality for exchanging solution vector data across subgrid
// rows via point-to-point messages.
//

// FIXME the whole exchange_results partnering process is silly and
// overengineered.
/* utility class for round robin selection, used in dependencies_to_messages */
class round_robin_wheel
{
public:
  round_robin_wheel(int const size) : size(size), current_index(0) {}

  int spin()
  {
    int const n = current_index++;

    if (current_index == size)
      current_index = 0;

    return n;
  }

private:
  int const size;
  int current_index;
};

/* this function takes the dependencies for each subgrid column,
 * and matches specific subgrid column members with the subgrid row
 * members that have the data they need in a balanced fashion.
 *
 * return vector is a list of message lists, one for each rank,
 * indexed by rank number */
std::vector<std::vector<message>> const static dependencies_to_messages(
    std::vector<rows_to_range> const &col_dependencies,
    std::vector<int> const &row_boundaries,
    std::vector<int> const &column_boundaries)
{
  expect(col_dependencies.size() == column_boundaries.size());

  /* initialize a round robin selector for each row */
  std::vector<round_robin_wheel> row_round_robin_wheels;
  for (int i = 0; i < static_cast<int>(row_boundaries.size()); ++i)
  {
    row_round_robin_wheels.emplace_back(column_boundaries.size());
  }

  /* this vector contains lists of messages indexed by rank */
  std::vector<std::vector<message>> messages(row_boundaries.size() *
                                             column_boundaries.size());

  /* iterate over each subgrid column's input requirements */
  for (int c = 0; c < static_cast<int>(col_dependencies.size()); c++)
  {
    /* dependencies describes the subgrid rows each column member will need
     * to communicate with, as well as the solution vector ranges needed
     * from each. these requirements are the same for every column member */
    rows_to_range const dependencies = col_dependencies[c];
    for (auto const &[row, limits] : dependencies)
    {
      /* iterate every rank in the subgrid column */
      for (int r = 0; r < static_cast<int>(row_boundaries.size()); ++r)
      {
        /* construct the receive item */
        int const receiver_rank = r * column_boundaries.size() + c;

        /* if receiver_rank has the data it needs locally, it will copy from its
         * own output otherwise, use round robin wheel to select a sender from
         * another row - every member of the row has the same data */
        int const sender_rank = [row = row, r, receiver_rank,
                                 &column_boundaries,
                                 &wheel = row_round_robin_wheels[row]]() {
          if (row == r)
          {
            return receiver_rank;
          }
          return static_cast<int>(row * column_boundaries.size() +
                                  wheel.spin());
        }();

        /* add message to the receiver's message list */
        message const incoming_message(message_direction::receive, sender_rank,
                                       limits);
        messages[receiver_rank].push_back(incoming_message);

        /* construct and enqeue the corresponding send item */
        message const outgoing_message(message_direction::send, receiver_rank,
                                       limits);
        messages[sender_rank].push_back(outgoing_message);
      }
    }
  }

  return messages;
}

/* generate_messages() creates a set of
   messages for each rank.*/

/* given a distribution plan, map each rank to a list of messages
 * index "x" of this vector contains the messages that must be transmitted
 * from and to rank "x" */

/* if the messages are invoked in the order they appear in the vector,
 * they are guaranteed not to produce a deadlock */
std::vector<std::vector<message>> const
generate_messages(distribution_plan const &plan)
{
  /* first, determine the subgrid tiling for this plan */
  std::vector<int> row_boundaries;
  std::vector<int> col_boundaries;

  auto const num_cols = get_num_subgrid_cols(plan.size());
  expect(plan.size() % num_cols == 0);
  auto const num_rows = static_cast<int>(plan.size()) / num_cols;

  for (int i = 0; i < num_rows; ++i)
  {
    element_subgrid const &grid = plan.at(i * num_cols);
    row_boundaries.push_back(grid.row_stop);
  }

  for (int i = 0; i < num_cols; ++i)
  {
    element_subgrid const &grid = plan.at(i);
    col_boundaries.push_back(grid.col_stop);
  }

  /* describe the rows/ranges each column needs to communicate with */
  auto const col_dependencies =
      find_column_dependencies(row_boundaries, col_boundaries);
  /* finally, build message list */
  auto const messages = dependencies_to_messages(
      col_dependencies, row_boundaries, col_boundaries);

  return messages;
}

// static helper for copying my own output to input
// message ranges are in terms of global element indices
// index maps get us back to local element indices
template<typename P>
static void
copy_to_input(fk::vector<P> const &source, fk::vector<P> &dest,
              index_mapper const &source_map, index_mapper const &dest_map,
              message const &message, int const segment_size)
{
  expect(segment_size > 0);
  if (message.message_dir == message_direction::send)
  {
    auto const source_start =
        static_cast<int64_t>(source_map(message.source_range.start)) *
        segment_size;
    auto const source_end =
        static_cast<int64_t>(source_map(message.source_range.stop) + 1) *
            segment_size -
        1;
    auto const dest_start =
        static_cast<int64_t>(dest_map(message.dest_range.start)) * segment_size;
    auto const dest_end =
        static_cast<int64_t>(dest_map(message.dest_range.stop) + 1) *
            segment_size -
        1;

    fk::vector<P, mem_type::const_view> const source_window(
        source, source_start, source_end);
    fk::vector<P, mem_type::view> dest_window(dest, dest_start, dest_end);

    fm::copy(source_window, dest_window);
  }
  // else ignore the matching receive; I am copying locally
}

// static helper for sending/receiving output/input data using mpi
// message ranges are in terms of global element indices
// index map gets us back to local element indices
template<typename P>
static void dispatch_message(fk::vector<P> const &source, fk::vector<P> &dest,
                             index_mapper const &map, message const &message,
                             int const segment_size)
{
#ifdef ASGARD_USE_MPI
  expect(segment_size > 0);

  MPI_Datatype const mpi_type =
      std::is_same<P, double>::value ? MPI_DOUBLE : MPI_FLOAT;
  MPI_Comm const communicator = distro_handle.get_global_comm();

  auto const mpi_tag = 0;
  if (message.message_dir == message_direction::send)
  {
    auto const source_start =
        static_cast<int64_t>(map(message.source_range.start)) * segment_size;
    auto const source_end =
        static_cast<int64_t>(map(message.source_range.stop) + 1) *
            segment_size -
        1;

    fk::vector<P, mem_type::const_view> const window(source, source_start,
                                                     source_end);

    auto const success =
        MPI_Send((void *)window.data(), window.size(), mpi_type, message.target,
                 mpi_tag, communicator);
    expect(success == 0);
  }
  else
  {
    auto const dest_start =
        static_cast<int64_t>(map(message.dest_range.start)) * segment_size;
    auto const dest_end =
        static_cast<int64_t>(map(message.dest_range.stop) + 1) * segment_size -
        1;

    fk::vector<P, mem_type::view> window(dest, dest_start, dest_end);

    auto const success =
        MPI_Recv((void *)window.data(), window.size(), mpi_type, message.target,
                 MPI_ANY_TAG, communicator, MPI_STATUS_IGNORE);
    expect(success == 0);
  }
#else

  ignore(source);
  ignore(dest);
  ignore(map);
  ignore(message);
  ignore(segment_size);
  expect(false);

#endif
}

template<typename P>
void exchange_results(fk::vector<P> const &source, fk::vector<P> &dest,
                      int const segment_size, distribution_plan const &plan,
                      int const my_rank)
{
  expect(my_rank >= 0);
  expect(my_rank < static_cast<int>(plan.size()));
#ifdef ASGARD_USE_MPI

  if (plan.size() == 1)
  {
    fm::copy(source, dest);
    return;
  }

  // build communication plan
  auto const message_lists = generate_messages(plan);

  // call send/recv
  auto const &my_subgrid = plan.at(my_rank);
  auto const messages    = message_lists[my_rank];

  for (auto const &message : messages)
  {
    if (message.target == my_rank)
    {
      copy_to_input(source, dest, my_subgrid.get_local_row_map(),
                    my_subgrid.get_local_col_map(), message, segment_size);
      continue;
    }

    auto const local_map = (message.message_dir == message_direction::send)
                               ? my_subgrid.get_local_row_map()
                               : my_subgrid.get_local_col_map();
    dispatch_message(source, dest, local_map, message, segment_size);
  }

#else
  ignore(segment_size);
  fm::copy(source, dest);
  return;
#endif
}

// gather errors from other local ranks
// returns {rmse errors, relative errors}
template<typename P>
std::array<fk::vector<P>, 2>
gather_errors(P const root_mean_squared, P const relative)
{
#ifdef ASGARD_USE_MPI

  std::array<P, 2> const error{root_mean_squared, relative};
  MPI_Comm local_comm;
  auto success =
      MPI_Comm_split_type(distro_handle.get_global_comm(), MPI_COMM_TYPE_SHARED,
                          0, MPI_INFO_NULL, &local_comm);
  expect(success == 0);

  MPI_Datatype const mpi_type =
      std::is_same<P, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  int local_rank;
  success = MPI_Comm_rank(local_comm, &local_rank);
  expect(success == 0);
  int local_size;
  success = MPI_Comm_size(local_comm, &local_size);
  expect(success == 0);

  fk::vector<P> error_vect(local_size * 2);

  MPI_Gather((void *)&error[0], 2, mpi_type, (void *)error_vect.data(), 2,
             mpi_type, 0, local_comm);

  success = MPI_Comm_free(&local_comm);
  expect(success == 0);

  if (local_rank == 0)
  {
    bool odd = false;
    std::vector<P> rmse, relative;
    // split the even and odd elements into seperate vectors -
    // unpackage from MPI call
    std::partition_copy(error_vect.begin(), error_vect.end(),
                        std::back_inserter(rmse), std::back_inserter(relative),
                        [&odd](P) { return odd = !odd; });
    return {fk::vector<P>(rmse), fk::vector<P>(relative)};
  }

  return {fk::vector<P>{root_mean_squared}, fk::vector<P>{relative}};
#else
  return {fk::vector<P>{root_mean_squared}, fk::vector<P>{relative}};
#endif
}

template<typename P>
std::vector<P>
gather_results(fk::vector<P> const &my_results, distribution_plan const &plan,
               int const my_rank, int const element_segment_size)
{
  expect(my_rank >= 0);
  expect(my_rank < static_cast<int>(plan.size()));

  auto const own_results = [&my_results]() {
    std::vector<P> own_results(my_results.size());
    std::copy(my_results.begin(), my_results.end(), own_results.begin());
    return own_results;
  };
#ifdef ASGARD_USE_MPI

  if (plan.size() == 1)
  {
    return own_results();
  }

  int const num_subgrid_cols = get_num_subgrid_cols(plan.size());

  // get the length and displacement of non-root, first row ranks
  fk::vector<int> const rank_lengths = [&plan, num_subgrid_cols,
                                        element_segment_size]() {
    fk::vector<int> rank_lengths(num_subgrid_cols);
    for (int i = 1; i < static_cast<int>(rank_lengths.size()); ++i)
    {
      rank_lengths(i) = plan.at(i).ncols() * element_segment_size;
    }
    return rank_lengths;
  }();

  fk::vector<int> const rank_displacements = [&rank_lengths]() {
    fk::vector<int> rank_displacements(rank_lengths.size());

    int64_t running_total = 0;
    for (int i = 0; i < rank_lengths.size(); ++i)
    {
      rank_displacements(i) = running_total;
      running_total += rank_lengths(i);
    }
    return rank_displacements;
  }();

  // split the communicator - only need the first row
  auto const participating            = my_rank < num_subgrid_cols;
  auto const comm_color               = participating ? 1 : MPI_UNDEFINED;
  MPI_Comm const &global_communicator = distro_handle.get_global_comm();
  MPI_Comm first_row_communicator;
  auto success = MPI_Comm_split(global_communicator, comm_color, my_rank,
                                &first_row_communicator);
  expect(success == 0);

  // gather values
  if (first_row_communicator != MPI_COMM_NULL)
  {
    int64_t const vect_size =
        my_rank ? 0
                : std::accumulate(rank_lengths.begin(), rank_lengths.end(),
                                  my_results.size());
    std::vector<P> results(vect_size);

    MPI_Datatype const mpi_type =
        std::is_same<P, double>::value ? MPI_DOUBLE : MPI_FLOAT;

    if (my_rank == 0)
    {
      std::copy(my_results.begin(), my_results.end(), results.begin());

      for (auto i = 1; i < num_subgrid_cols; ++i)
      {
        success = MPI_Recv((void *)(results.data() + my_results.size() +
                                    rank_displacements(i)),
                           rank_lengths(i), mpi_type, i, MPI_ANY_TAG,
                           first_row_communicator, MPI_STATUS_IGNORE);
        expect(success == 0);
      }

      return results;
    }
    else
    {
      auto const mpi_tag = 0;
      success = MPI_Send((void *)my_results.data(), my_results.size(), mpi_type,
                         0, mpi_tag, first_row_communicator);

      expect(success == 0);
      return own_results();
    }
  }

  return own_results();

#else
  ignore(element_segment_size);
  return own_results();
#endif
}

template<typename P>
P get_global_max(P const my_max, distribution_plan const &plan)
{
#ifdef ASGARD_USE_MPI

  // split into rows
  MPI_Comm row_communicator;
  MPI_Comm const global_communicator = distro_handle.get_global_comm();
  auto const num_cols                = get_num_subgrid_cols(plan.size());
  auto const my_rank                 = get_rank();
  auto const my_row                  = my_rank / num_cols;
  auto const my_col                  = my_rank % num_cols;
  auto success =
      MPI_Comm_split(global_communicator, my_row, my_col, &row_communicator);
  assert(success == 0);

  // get max
  MPI_Datatype const mpi_type =
      std::is_same<P, double>::value ? MPI_DOUBLE : MPI_FLOAT;

  P global_max;
  success = MPI_Allreduce(&my_max, &global_max, 1, mpi_type, MPI_MAX,
                          row_communicator);
  assert(success == 0);
  success = MPI_Comm_free(&row_communicator);
  assert(success == 0);

#else
  assert(plan.size() == 1);
  auto const global_max = my_max;
#endif

  return global_max;
}

std::vector<int64_t>
distribute_table_changes(std::vector<int64_t> const &my_changes,
                         distribution_plan const &plan)
{
  if (plan.size() == 1)
  {
    return my_changes;
  }

#ifdef ASGARD_USE_MPI
  // determine size of everyone's messages
  auto const my_rank      = get_rank();
  auto const num_messages = [&plan, &my_changes, my_rank]() {
    std::vector<int> num_messages(plan.size());
    assert(my_changes.size() < INT_MAX);
    num_messages[get_rank()] = static_cast<int>(my_changes.size());
    assert(plan.size() < INT_MAX);
    for (auto i = 0; i < static_cast<int>(plan.size()); ++i)
    {
      auto const success = MPI_Bcast(num_messages.data() + i, 1, MPI_INT, i,
                                     distro_handle.get_global_comm());
      assert(success == 0);
    }
    return num_messages;
  }();

  auto const displacements = [&num_messages]() {
    std::vector<int> displacements(num_messages.size());

    // some compilers don't support this yet...//FIXME
    // std::exclusive_scan(num_messages.begin(), num_messages.end(),
    //                    displacements.begin(), 0);
    // do it manually instead...
    for (auto i = 1; i < static_cast<int>(num_messages.size()); ++i)
    {
      displacements[i] = displacements[i - 1] + num_messages[i - 1];
    }
    return displacements;
  }();

  // now distribute changes
  std::vector<int64_t> all_changes(
      std::accumulate(num_messages.begin(), num_messages.end(), 0));

  auto const success = MPI_Allgatherv(
      my_changes.data(), static_cast<int>(my_changes.size()), MPI_INT64_T,
      all_changes.data(), num_messages.data(), displacements.data(),
      MPI_INT64_T, distro_handle.get_global_comm());

  assert(success == 0);

  return all_changes;

#else
  // plan size > 1, MPI off - shouldn't occur
  assert(false);
  return my_changes;
#endif
}

// private helper.
// find the rank on my subgrid row that has the old x data new_region needs,
// using old plan
std::unordered_map<int, std::list<message>>
region_messages_remap(distribution_plan const &old_plan,
                      distribution_plan const &new_plan,
                      grid_limits const source_region,
                      grid_limits const dest_region, int const rank)
{
  assert(rank >= 0);

  // we should only need to communicate with our own (old) row of subgrids
  // each has a full copy of the vector
  auto const cols = get_num_subgrid_cols(old_plan.size());
  auto const row  = rank / cols;

  std::unordered_map<int, std::list<message>> region_messages;
  auto const queue_msg = [&region_messages](int const rank,
                                            message const &msg) {
    region_messages.try_emplace(rank, std::list<message>());
    region_messages[rank].push_back(msg);
  };

  // iterate over columns in the subgrid
  // find grids that include parts (or all)
  // of this region.
  auto const my_new_subgrid = new_plan.at(rank);
  auto dest_subregion_start = dest_region.start;
  assert(dest_subregion_start >= my_new_subgrid.col_start);

  for (auto i = 0; i < cols; ++i)
  {
    // FIXME assumes a row major layout of the element grid
    auto const old_source_subgrid = old_plan.at(i);

    // if the source region is contained within the old remote subgrid
    if (source_region.start <= old_source_subgrid.col_stop &&
        source_region.stop >= old_source_subgrid.col_start)
    {
      // find source values contained within the remote subgrid
      auto const contain_start =
          std::max(source_region.start, old_source_subgrid.col_start);
      auto const contain_end =
          std::min(source_region.stop, old_source_subgrid.col_stop);
      auto const contain_subregion = grid_limits(contain_start, contain_end);

      // where I am going to place those values in this subgrid
      auto const dest_subregion_stop =
          std::min(dest_subregion_start + contain_subregion.size() - 1,
                   my_new_subgrid.col_stop);
      auto const dest_subregion =
          grid_limits(dest_subregion_start, dest_subregion_stop);
      dest_subregion_start = dest_subregion_stop + 1;

      // trim to fit dest
      auto const source_subregion =
          grid_limits(contain_subregion.start,
                      contain_subregion.start + dest_subregion.size() - 1);

      // enqueue the communication with
      // the subgrid at old_grid(rank_row, i)
      auto const partner_rank = row * cols + i;

      // no deadlock ensured by simultaneously enqueing the recv and send
      // receive
      queue_msg(rank, message(message_direction::receive, partner_rank,
                              source_subregion, dest_subregion));
      // matching send
      queue_msg(partner_rank, message(message_direction::send, rank,
                                      source_subregion, dest_subregion));
    }
  }

  assert(dest_subregion_start = dest_region.stop + 1);
  return region_messages;
}

// private helper.
// map to the old x data my subgrid requires,
// using the element remapping
static std::vector<std::list<message>>
subgrid_messages_remap(distribution_plan const &old_plan,
                       distribution_plan const &new_plan,
                       std::map<int64_t, grid_limits> const &elem_index_remap,
                       int const rank)
{
  assert(rank >= 0);

  std::vector<std::list<message>> subgrid_messages(new_plan.size());
  auto const rank_subgrid = new_plan.at(rank);

  for (auto const &[new_index_start, old_region] : elem_index_remap)
  {
    // containment test
    auto const new_index_stop = new_index_start + old_region.size() - 1;
    if (new_index_start > rank_subgrid.col_stop ||
        new_index_stop < rank_subgrid.col_start)
    {
      continue;
    }

    auto const dest_start =
        std::max(new_index_start, static_cast<int64_t>(rank_subgrid.col_start));
    auto const dest_stop =
        std::min(new_index_stop, static_cast<int64_t>(rank_subgrid.col_stop));
    auto const dest_region = grid_limits(dest_start, dest_stop);
    auto const source_start =
        std::max(static_cast<int64_t>(old_region.start),
                 old_region.start + (rank_subgrid.col_start - new_index_start));
    auto const source_stop =
        std::min(static_cast<int64_t>(old_region.stop),
                 old_region.stop - (new_index_stop - rank_subgrid.col_stop));
    auto const source_region = grid_limits(source_start, source_stop);

    auto region_messages = region_messages_remap(
        old_plan, new_plan, source_region, dest_region, rank);

    for (auto &[rank, msgs] : region_messages)
    {
      subgrid_messages[rank].splice(subgrid_messages[rank].end(), msgs);
    }
  }
  return subgrid_messages;
}

std::vector<std::list<message>>
generate_messages_remap(distribution_plan const &old_plan,
                        distribution_plan const &new_plan,
                        std::map<int64_t, grid_limits> const &elem_index_remap)
{
  assert(old_plan.size() == new_plan.size());
  assert(elem_index_remap.size() != 0);

  // TODO technically, all these calculations will yield the same set of
  // messages for each subgrid row. if this requires optimization, just perform
  // the first row, and add functionality to replicate messaging behavior across
  // rows
  std::vector<std::list<message>> redis_messages(new_plan.size());
  for (auto const &[rank, subgrid] : new_plan)
  {
    ignore(subgrid);
    auto rank_messages =
        subgrid_messages_remap(old_plan, new_plan, elem_index_remap, rank);
    assert(rank_messages.size() == new_plan.size());
    for (auto const &[rank, subgrid] : new_plan)
    {
      ignore(subgrid);
      redis_messages[rank].splice(redis_messages[rank].end(),
                                  rank_messages[rank]);
    }
  }
  return redis_messages;
}

static bool check_overlap(std::map<int64_t, grid_limits> const &elem_remap)
{
  auto next_valid = 0;
  for (auto const &[new_index, old_region] : elem_remap)
  {
    if (new_index < next_valid)
    {
      return false;
    }
    next_valid = new_index + old_region.size();
  }
  return true;
}

template<typename P>
fk::vector<P>
redistribute_vector(fk::vector<P> const &old_x,
                    distribution_plan const &old_plan,
                    distribution_plan const &new_plan,
                    std::map<int64_t, grid_limits> const &elem_remap)
{
  assert(old_plan.size() == new_plan.size());
  assert(check_overlap(elem_remap));
  assert(elem_remap.size() != 0);

  auto const my_rank     = get_rank();
  auto const old_subgrid = old_plan.at(my_rank);
  auto const new_subgrid = new_plan.at(my_rank);

  // x's size should be num_elements*deg^dim
  // (deg^dim is one element's number of coefficients/ elements in x vector)
  assert(old_x.size() % old_subgrid.ncols() == 0);
  auto const segment_size = old_x.size() / old_subgrid.ncols();
  fk::vector<P> y(new_subgrid.ncols() * segment_size);

  auto const messages =
      generate_messages_remap(old_plan, new_plan, elem_remap)[my_rank];

  for (auto const &message : messages)
  {
    auto const source_local_map = old_subgrid.get_local_col_map();
    auto const dest_local_map   = new_subgrid.get_local_col_map();
    if (message.target == my_rank)
    {
      copy_to_input(old_x, y, source_local_map, dest_local_map, message,
                    segment_size);
    }
    else
    {
      auto const local_map = (message.message_dir == message_direction::send)
                                 ? source_local_map
                                 : dest_local_map;
      dispatch_message(old_x, y, local_map, message, segment_size);
    }
  }
  return y;
}

#define X(T)                                            \
  template void reduce_results(                         \
      fk::vector<T> const &source, fk::vector<T> &dest, \
      distribution_plan const &plan, int const my_rank);
#include "type_list_float.inc"
#undef X

#define X(T)                                                                  \
  template void exchange_results(fk::vector<T> const &source,                 \
                                 fk::vector<T> &dest, int const segment_size, \
                                 distribution_plan const &plan,               \
                                 int const my_rank);
#include "type_list_float.inc"
#undef X

#define X(T)                                           \
  template std::array<fk::vector<T>, 2> gather_errors( \
      T const root_mean_squared, T const relative);
#include "type_list_float.inc"
#undef X

#define X(T)                                                          \
  template std::vector<T> gather_results(                             \
      fk::vector<T> const &my_results, distribution_plan const &plan, \
      int const my_rank, int const element_segment_size);
#include "type_list_float.inc"
#undef X

#define X(T) \
  template T get_global_max(T const my_max, distribution_plan const &plan);
#include "type_list_float.inc"
#undef X

#define X(T)                                                         \
  template fk::vector<T> redistribute_vector(                        \
      fk::vector<T> const &old_x, distribution_plan const &old_plan, \
      distribution_plan const &new_plan,                             \
      std::map<int64_t, grid_limits> const &elem_remap);
#include "type_list_float.inc"
#undef X
