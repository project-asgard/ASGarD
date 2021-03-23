#include "tools.hpp"
#include <algorithm>
// FIXME use string format after C++20
// #include <format>
#include <math.h>
#include <numeric>
#include <sstream>
#include <timemory/library.h>
#include <timemory/timemory.hpp>
#include <timemory/tools/timemory-mpip.h>
// TODO make an initialize and finalize function

namespace tools
{
std::string simple_timer::report()
{
  std::ostringstream report;
  report << "\nperformance report, all times in ms...\n\n";

  for (auto [id, times] : id_to_times_)
  {
    auto const avg =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    auto const min = *std::min_element(times.begin(), times.end());
    auto const max = *std::max_element(times.begin(), times.end());

    // calculate median
    auto const middle_it = times.begin() + times.size() / 2;
    std::nth_element(times.begin(), middle_it, times.end());
    auto const med =
        times.size() % 2 == 0
            ? (*std::max_element(times.begin(), middle_it) + *middle_it) / 2
            : *middle_it;

    auto const avg_flops = [this, id = id]() {
      if (id_to_flops_.count(id) > 0)
      {
        auto const flops = id_to_flops_[id];
        auto const sum   = std::accumulate(flops.begin(), flops.end(), 0.0);

        if (isinf(sum))
        {
          return std::string(" avg gflops: inf");
        }
        auto const avg = sum / flops.size();
        return std::string(" avg gflops: ") + std::to_string(avg);
      }
      return std::string("");
    }();

    report << id << " - avg: " << avg << " min: " << min << " max: " << max
           << " med: " << med << avg_flops << " calls: " << times.size()
           << '\n';
  }
  return report.str();
}

simple_timer timer;
} // namespace tools

void start(const char *name)
{
  // use library or template API
  timemory_push_region(name);
}
void stop(const char *name)
{
  // use library or template API
  timemory_pop_region(name);
}

namespace profiling
{
void begin_iteration(const std::string &_name);

// data-size is some normalization value for int[10] vs. double[20], etc.
void end_iteration(const std::string &_name, int64_t _data_size,
                   int64_t _num_itr);
} // namespace profiling

// TIMEMORY_DEFINE_API(asgard);
namespace tim
{
namespace api
{
struct asgard : concepts::api
{};
} // namespace api
} // namespace tim

using namespace tim::component;
using namespace tim::concepts;
using namespace tim::component::operators;
using iteration_counter = data_tracker<int64_t, tim::api::asgard>;

TIMEMORY_DECLARE_COMPONENT(iteration_tracker)

namespace tim
{
namespace component
{
struct iteration_profiler : base<iteration_profiler, double>
{
  // TODO add member functuion to print for pair
  // get_display
  static std::string label() { return "asgard_iteration_profiler"; }
  static std::string description() { return "normalized iteration data"; }

  void start() { m_papi.start(); }

  void stop(int64_t data_size, int64_t nitr)
  {
    m_papi.stop();

    // raw flops and load/stores
    long long flops = m_papi.get_value().at(0) + m_papi.get_value().at(1);
    // long long ldst = m_papi.get_value().at(2);

    // normalized flops per iteration
    value /*.first*/ = static_cast<double>(flops) / data_size / nitr;
    // normalized load/stores per iteration
    // value.second = static_cast<double>(ldst) / data_size / nitr;

    // when structure is reused, value is last measurement, accum is sum
    // std::tie(accum.first, accum.second) += std::tie(value.first,
    // value.second);
    accum += value;
  }

private:
  // papi_tuple<PAPI_DP_OPS, PAPI_SP_OPS/*, PAPI_LST_INS*/> m_papi;
  papi_tuple<PAPI_LD_INS, PAPI_SR_INS> m_papi;
};
} // namespace component
} // namespace tim
// create a bundle with wall-clock timer (always on), two custom components,
// and user_global_bundle for runtime insertion of other components
// auto_bundle bugged --> component_bundle.
using iteration_bundle_t =
    tim::component_bundle<tim::api::asgard, // constructe cals start no argument
                          iteration_counter, iteration_profiler>;

using iteration_data_t = std::unordered_map<std::string, iteration_bundle_t>;

static iteration_data_t iteration_data;

namespace profiling
{
void start(const std::string &_name) { timemory_push_region(_name.c_str()); }
void stop(const std::string &_name) { timemory_pop_region(_name.c_str()); }

void begin_iteration(const std::string &_name)
{
  std::cerr << "START ITERATION" << std::endl;
  // iteration_data.emplace(_name, iteration_bundle_t{ _name });
  iteration_data.emplace(_name, iteration_bundle_t{_name})
      .first->second.start();
  start(_name);
}

void end_iteration(const std::string &_name, int64_t _data_size,
                   int64_t _num_itr)
{
  stop(_name);
  auto itr = iteration_data.find(_name);
  if (itr == iteration_data.end())
    return;

  // this will add num iterations to data tracker
  itr->second.store(_num_itr);
  // create a child entry in data-tracker call-graph with the name "normalized".
  itr->second.add_secondary("normalized", _num_itr / _data_size);
  // calls stop on all components and passes data-size/iterations to
  // iteration_profiler
  itr->second.stop(_data_size, _num_itr);
  // itr->second.stop();
  // erase the entry
  iteration_data.erase(itr);
}
} // namespace profiling

/*
   using namespace tim::component;
   struct asgard_profiler {}; // dummy

   using iteration_counter = data_tracker<int64_t, asgard_profiler>;

   struct iteration_profiler : base<iteration_profiler, std::pair<double,
double>>
   {

   void start(int64_t _data_size)
   {
   m_papi.start();
   }

   void stop(int64_t data_size, int64_t nitr)
   {
   m_papi.stop();

// raw flops and load/stores
long long flops = m_papi.get_value().at(0) + m_papi.get_value().at(1);
long long ldst = m_papi.get_value().at(2);

// normalized flops per iteration
value.first = static_cast<double>(flops) / data_size / nitr;
// normalized load/stores per iteration
value.second = static_cast<double>(ldst) / data_size / nitr;

// when structure is reused, value is last measurement, accum is sum
std::tie(accum.first, accum.second) += std::tie(value.first, value.second);
}

private:
papi_tuple<PAPI_DP_OPS, PAPI_SP_OPS, PAPI_LST_INS> m_papi;
};

// create a bundle with wall-clock timer (always on), two custom components,
// and user_global_bundle for runtime insertion of other components
using iteration_bundle_t = tim::auto_bundle<asgard_profiler, wall_clock,
iteration_counter, iteration_profiler,
user_global_bundle>;

using iteration_data_t = std::unordered_map<std::string, iteration_bundle_t>;

static iteration_data_t iteration_data;

void begin_iteration(const std::string& _name)
{
iteration_data.emplace(_name, iteration_bundle_t{ _name });
}

void end_iteration(const std::string& _name, int64_t _data_size, int64_t
_num_itr)
{
auto itr = iteration_data.at(_name);
if(itr == iteration_data.end())
return;

// this will add num iterations to data tracker
itr->store(_num_itr);
// create a child entry in data-tracker call-graph with the name "normalized".
itr->add_secondary("normalized", _num_itr / _data_size);
// calls stop on all components and passes data-size/iterations to
iteration_profiler itr->stop(_data_size, _num_itr);
// erase the entry
iteration_data.erase(itr);
}

} // namespace tools
*/
