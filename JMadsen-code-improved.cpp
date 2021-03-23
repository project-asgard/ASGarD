#include <type_traits>
#include <tuple>
#include <memory>
#include <iostream>

template <typename P>
class PDE
{};

template <typename P>
class PDE_continuity_1d : public PDE<P>
{
    public:
        PDE_continuity_1d() { puts(__FUNCTION__); }
};

template <typename P>
class PDE_continuity_2d : public PDE<P>
{
    public:
        PDE_continuity_2d() { puts(__FUNCTION__); }
};

enum class PDE_opts
{
    continuity_1 = 0,
    continuity_2,
    //    continuity_3,
    //    continuity_6,
    //    fokkerplanck_1d_4p1a,
    //    fokkerplanck_1d_4p2,
    //    fokkerplanck_1d_4p3,
    //    fokkerplanck_1d_4p4,
    //    fokkerplanck_1d_4p5,
    //    fokkerplanck_2d_complete,
    //    diffusion_1,
    //    diffusion_2,
    // FIXME will need to add the user supplied PDE choice
};

struct parser
{
    PDE_opts get_selected_pde() const { return PDE_opts::continuity_2; }
};


// this just holds types
template <template <typename> class... T>
struct type_list
{};

namespace traits
{
    // use this to map a type to an enumerator
    template <template <typename> class T>
        struct enumerator;
}

// macro to specialize
#define ENUM_TRAIT(TYPE, ENUM)                                                           \
    namespace traits                                                                     \
{                                                                                    \
    template <>                                                                          \
    struct enumerator<TYPE>                                                              \
    {                                                                                    \
        static constexpr auto value = ENUM;                                              \
        template <typename U>                                                            \
        using type = TYPE<U>;                                                            \
    };                                                                                   \
}

    ENUM_TRAIT(PDE_continuity_1d, PDE_opts::continuity_1)
ENUM_TRAIT(PDE_continuity_2d, PDE_opts::continuity_2)
    // ... etc. ...


    namespace impl
{
    template <typename P, template <typename> class T, template <typename> class... Tail>
        std::unique_ptr<PDE<P>>
        make_PDE(PDE_opts _opt, type_list<T, Tail...>,
                std::enable_if_t<(sizeof...(Tail) == 0), int> = 0)
        {
            using enumerator = traits::enumerator<T>;
            if(enumerator::value == _opt)
                return std::make_unique<typename enumerator::template type<P>>();
            std::cout << "Invalid pde choice" << std::endl;
            exit(-1);
        }

    template <typename P, template <typename> class T, template <typename> class... Tail>
        std::unique_ptr<PDE<P>>
        make_PDE(PDE_opts _opt, type_list<T, Tail...>,
                std::enable_if_t<(sizeof...(Tail) > 0), int> = 0)
        {
            using enumerator = traits::enumerator<T>;
            if(enumerator::value == _opt)
                return std::make_unique<typename enumerator::template type<P>>();
            return make_PDE<P>(_opt, type_list<Tail...>{});
        }
}

    template <typename P>
std::unique_ptr<PDE<P>> make_PDE(struct parser const& cli_input)
{
    return impl::make_PDE<P>(cli_input.get_selected_pde(), type_list<PDE_continuity_1d, PDE_continuity_2d>{});
}

int main()
{
    parser _parser{};
    auto _val = make_PDE<float>(_parser);
}

