#include "adapt.hpp"
#include "distribution.hpp"
#include "transformations.hpp"

#include <unordered_set>

namespace adapt
{
static std::map<int64_t, grid_limits>
remap_for_addtl(int64_t const old_num_elems)
{
  assert(old_num_elems > 0);
  std::map<int64_t, grid_limits> mapper;
  // beginning of new elem range maps directly to old elem range
  mapper.insert({0, grid_limits(0, old_num_elems - 1)});
  return mapper;
}

static std::map<int64_t, grid_limits>
remap_for_delete(std::vector<int64_t> const &deleted_indices,
                 int64_t const num_new_elems)
{
  assert(num_new_elems > 0);

  if (deleted_indices.size() == 0)
  {
    return {};
  }

  std::unordered_set<int64_t> deleted(deleted_indices.begin(),
                                      deleted_indices.end());

  std::map<int64_t, grid_limits> new_to_old;

  int64_t old_index    = 0;
  int64_t new_index    = 0;
  int64_t retain_count = 0;

  while (new_index < num_new_elems)
  {
    // while in a preserved region, advance both indices
    // count how many elements are in the region
    while (deleted.count(old_index) == 0)
    {
      old_index++;
      new_index++;
      retain_count++;
      if (new_index >= num_new_elems - 1)
      {
        break;
      }
    }

    // record preserved region
    if (retain_count > 0)
    {
      new_to_old.insert({new_index - retain_count,
                         grid_limits(old_index - retain_count, old_index - 1)});
      retain_count = 0;
    }

    // skip past deleted regions
    while (deleted.count(old_index) == 1)
    {
      old_index++;
    }
  }

  return new_to_old;
}

// helper to find new levels for each dimension after adapting table
static std::vector<int>
get_levels(elements::table const &adapted_table, int const num_dims)
{
  assert(num_dims > 0);
  auto const flat_table = adapted_table.get_active_table().clone_onto_host();
  auto const coord_size = num_dims * 2;
  std::vector<int> max_levels(num_dims, 0);
  for (int64_t element = 0; element < adapted_table.size(); ++element)
  {
    fk::vector<int, mem_type::const_view> coords(
        flat_table, element * coord_size, (element + 1) * coord_size - 1);
    for (auto i = 0; i < num_dims; ++i)
    {
      max_levels[i] = std::max(coords(i), max_levels[i]);
    }
  }
  return max_levels;
}

template<typename P>
static void update_levels(elements::table const &adapted_table, PDE<P> &pde,
                          bool const rechain = false)
{
  auto const new_levels =
      get_levels(adapted_table, pde.get_dimensions().size());
  for (auto i = 0; i < static_cast<int>(new_levels.size()); ++i)
  {
    pde.update_dimension(i, new_levels[i]);
    if (rechain)
    {
      pde.rechain_dimension(i);
    }
  }
}

template<typename P>
distributed_grid<P>::distributed_grid(PDE<P> const &pde,
                                      options const &cli_opts)
    : table_(cli_opts, pde)
{
  plan_ = get_plan(get_num_ranks(), table_);
}

// FIXME assumes uniform degree across levels
template<typename P>
fk::vector<P> distributed_grid<P>::get_initial_condition(
    PDE<P> &pde, basis::wavelet_transform<P, resource::host> const &transformer,
    options const &cli_opts)
{
  // get unrefined condition
  std::vector<vector_func<P>> v_functions;
  for (auto const &dim : pde.get_dimensions())
  {
    v_functions.push_back(dim.initial_condition);
  }
  P const time             = 0;
  auto const initial_unref = [this, &v_functions, &pde, &transformer, time]() {
    auto const subgrid = this->get_subgrid(get_rank());
    // TODO temp add scalar time func to initial conditions with multi-D func PR
    auto const mult =
        pde.has_analytic_soln ? pde.exact_time(time) : static_cast<P>(1.0);
    return transform_and_combine_dimensions(
        pde, v_functions, this->get_table(), transformer, subgrid.col_start,
        subgrid.col_stop, pde.get_dimensions()[0].get_degree(), time, mult);
  }();

  if (!cli_opts.do_adapt_levels)
  {
    return initial_unref;
  }

  // refine
  fk::vector<P> refine_y(initial_unref);
  auto refining = true;
  while (refining)
  {
    auto const old_y   = fk::vector<P>(refine_y);
    auto const refined = this->refine(old_y, cli_opts);
    refining           = old_y.size() != refined.size();
    update_levels(this->get_table(), pde);

    // reproject
    auto const reprojected = [this, &v_functions, &pde, &transformer, time]() {
      auto const subgrid = this->get_subgrid(get_rank());
      auto const mult =
          pde.has_analytic_soln ? pde.exact_time(time) : static_cast<P>(1.0);
      return transform_and_combine_dimensions(
          pde, v_functions, this->get_table(), transformer, subgrid.col_start,
          subgrid.col_stop, pde.get_dimensions()[0].get_degree(), time, mult);
    }();
    refine_y.resize(reprojected.size()) = reprojected;
  }

  // coarsen
  auto const coarse_y = this->coarsen(refine_y, cli_opts);
  update_levels(this->get_table(), pde);

  // reproject
  auto const adapted_y = [this, &v_functions, &pde, &transformer, time]() {
    auto const subgrid = this->get_subgrid(get_rank());
    auto const mult =
        pde.has_analytic_soln ? pde.exact_time(time) : static_cast<P>(1.0);
    return transform_and_combine_dimensions(
        pde, v_functions, this->get_table(), transformer, subgrid.col_start,
        subgrid.col_stop, pde.get_dimensions()[0].get_degree(), time, mult);
  }();

  return adapted_y;
}

template<typename P>
fk::vector<P>
distributed_grid<P>::coarsen_solution(PDE<P> &pde, fk::vector<P> const &x,
                                      options const &cli_opts)
{
  auto const coarse_y = this->coarsen(x, cli_opts);
  auto const rechain  = true;
  update_levels(this->get_table(), pde, rechain);
  return coarse_y;
}

template<typename P>
fk::vector<P>
distributed_grid<P>::refine_solution(PDE<P> &pde, fk::vector<P> const &x,
                                     options const &cli_opts)
{
  auto const refine_y = this->refine(x, cli_opts);
  auto const rechain  = true;
  update_levels(this->get_table(), pde, rechain);
  return refine_y;
}

template<typename P>
fk::vector<P>
distributed_grid<P>::redistribute_solution(fk::vector<P> const &x,
                                           distribution_plan const &old_plan,
                                           int const old_size)
{
  return redistribute_vector(x, old_plan, plan_, remap_for_addtl(old_size));
}

template<typename P>
fk::vector<P>
distributed_grid<P>::refine(fk::vector<P> const &x, options const &cli_opts)
{
  auto const abs_compare = [](auto const a, auto const b) {
    return (std::abs(a) < std::abs(b));
  };
  auto const max_elem =
      std::abs(*std::max_element(x.begin(), x.end(), abs_compare));
  auto const global_max = get_global_max(max_elem, this->plan_);

  auto const refine_threshold = cli_opts.adapt_threshold * global_max;
  if (refine_threshold <= 0.0)
  {
    return x;
  }

  auto const refine_check =
      [refine_threshold, abs_compare](
          int64_t const, fk::vector<P, mem_type::const_view> const &element_x) {
        auto const max_elem =
            *std::max_element(element_x.begin(), element_x.end(), abs_compare);
        return std::abs(max_elem) >= refine_threshold;
      };
  auto const to_refine = filter_elements(refine_check, x);
  return this->refine_elements(to_refine, cli_opts, x);
}

template<typename P>
fk::vector<P>
distributed_grid<P>::coarsen(fk::vector<P> const &x, options const &cli_opts)
{
  auto const abs_compare = [](auto const a, auto const b) {
    return (std::abs(a) < std::abs(b));
  };
  auto const max_elem =
      std::abs(*std::max_element(x.begin(), x.end(), abs_compare));
  auto const global_max       = get_global_max(max_elem, this->plan_);
  auto const refine_threshold = cli_opts.adapt_threshold * global_max;
  if (refine_threshold <= 0.0)
  {
    return x;
  }

  auto const coarsen_threshold = refine_threshold * 0.1;
  auto const &table            = this->table_;
  auto const coarsen_check =
      [&table, coarsen_threshold,
       abs_compare](int64_t const elem_index,
                    fk::vector<P, mem_type::const_view> const &element_x) {
        auto const max_elem =
            *std::max_element(element_x.begin(), element_x.end(), abs_compare);
        auto const coords    = table.get_coords(elem_index);
        auto const min_level = *std::min_element(
            coords.begin(), coords.begin() + coords.size() / 2);
        return std::abs(max_elem) <= coarsen_threshold && min_level >= 1;
      };

  auto const to_coarsen = filter_elements(coarsen_check, x);
  return this->remove_elements(to_coarsen, x);
}

template<typename P>
fk::vector<P> distributed_grid<P>::refine_elements(
    std::vector<int64_t> const &indices_to_refine, options const &opts,
    fk::vector<P> const &x)
{
  std::list<int64_t> child_ids;
  for (auto const parent_index : indices_to_refine)
  {
    child_ids.splice(child_ids.end(),
                     table_.get_child_elements(parent_index, opts));
  }

  // need to preserve ordering for testing
  auto const get_unique = [](auto const &ids) {
    std::unordered_set<int64_t> ids_so_far;
    std::vector<int64_t> unique_ids;
    for (auto const id : ids)
    {
      if (ids_so_far.count(id) == 0)
      {
        unique_ids.push_back(id);
      }
      ids_so_far.insert(id);
    }
    return unique_ids;
  };

  auto const all_child_ids =
      get_unique(distribute_table_changes(get_unique(child_ids), plan_));

  if (all_child_ids.size() == 0)
  {
    return x;
  }

  auto const added    = table_.add_elements(all_child_ids, opts.max_level);
  auto const new_plan = get_plan(get_num_ranks(), table_);
  auto const remapper = remap_for_addtl(table_.size() - added);
  auto const y        = redistribute_vector(x, plan_, new_plan, remapper);
  plan_               = new_plan;

  return y;
}

template<typename P>
fk::vector<P> distributed_grid<P>::remove_elements(
    std::vector<int64_t> const &indices_to_remove, fk::vector<P> const &x)
{
  auto const all_remove_indices =
      distribute_table_changes(indices_to_remove, plan_);

  if (all_remove_indices.size() == 0)
  {
    return x;
  }

  table_.remove_elements(all_remove_indices);
  if (this->size() < get_num_ranks())
  {
    node_out() << "coarsened below number of ranks - can't handle this case yet"
               << '\n';
  }
  auto const new_plan = get_plan(get_num_ranks(), table_);
  auto const remapper = remap_for_delete(all_remove_indices, table_.size());
  auto const y        = redistribute_vector(x, plan_, new_plan, remapper);
  plan_               = new_plan;
  return y;
}

#define X(T) template class distributed_grid<T>;
#include "type_list_float.inc"
#undef X

} // namespace adapt
