#include "compat.hpp"
#include <atomic>
#include <string>
#include <sstream>
#include <ostream>
#include <type_traits>

#ifndef G6K_HYBRID_SLICER_H
    #error Do not include siever.inl directly
#endif

#ifndef COLLECT_STATISTICS_SLICER
    // #if defined ENABLE_EXTENDED_STATS_SLICER
    //     #define COLLECT_STATISTICS_SLICER 2
    #if defined ENABLE_STATS_SLICER
        #define COLLECT_STATISTICS_SLICER 1
    #else
        #define COLLECT_STATISTICS_SLICER 0
    #endif
#endif

/**
    Define macros COLLECT_STATISTICS_*:
    COLLECT_STATISTICS_* gives a level how fine-grained the statistics are to be collected.
    0 means no colection takes place at all.
    * is one of:
    XORPOPCNT_SLICER ( the sum of two following values )
    XORPOPCNT_SLICER_R ( xor-pop-count computations during the randomization phase )
    XORPOPCNT_SLICER_S ( xor-pop-count computations during the slicing phase )
    XORPOPCNT_PASS_SLICER ( how often these are "successful" )
    XORPOPCNT_PASS_SLICER_R ( how often these are "successful" during the randomization phase )
    XORPOPCNT_PASS_SLICER_S ( how often these are "successful" during the slicing phase )
    FULLSCPRODS_SLICER ( full scalar product computations inside the randomization and slicing phase of the slicer)
    FILTER_PASS_SLICER ( how often vectors pass our filters (for bucketing / filtered lists) )
    REDSUCCESS_SLICER ( how often we actually successfully (believe to) create a short vector in dbt )
    REPLACEMENTS_SLICER ( dbt replacements - slicing phase only )
    REPLACEMENTFAILURE_SLICER ( failures for various reasons - slicing phase only )
    COLLISIONS_SLICER ( hash collisions )
    REDS_DURING_RANDOMIZATION (how many times targerts were reduced in the randomization phase) SORTING_SLICER
    BUCKETS_SLICER (number of buckets considered) REMOVE!
    BUCKETS_OVERFLOW_MAX_SLICER (max number of vectors attempted to be inserted into a bucket)
    BUCKETS_OVERFLOW_COUNT_SLICER (cumulative number of bucket overflows)
    ITERCOUNT_SLICER (number of iterations during the last call of bdgl_like_sieve)
    RANDOMIZE_TRIALNUM_SLICER (cumulative number of trials in randomize_target_small_task)
*/

#ifndef COLLECT_STATISTICS_XORPOPCNT_SLICER
#define COLLECT_STATISTICS_XORPOPCNT_SLICER COLLECT_STATISTICS_SLICER
#endif


#ifndef COLLECT_STATISTICS_XORPOPCNT_PASS_SLICER
#define COLLECT_STATISTICS_XORPOPCNT_PASS_SLICER COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_FULLSCPRODS_SLICER
#define COLLECT_STATISTICS_FULLSCPRODS_SLICER COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_FILTER_PASS_SLICER
#define COLLECT_STATISTICS_FILTER_PASS_SLICER COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_REDSUCCESS_SLICER
#define COLLECT_STATISTICS_REDSUCCESS_SLICER COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_REPLACEMENTS_SLICER
#define COLLECT_STATISTICS_REPLACEMENTS_SLICER COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_COLLISIONS_SLICER
#define COLLECT_STATISTICS_COLLISIONS_SLICER COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_REDS_DURING_RANDOMIZATION
#define COLLECT_STATISTICS_REDS_DURING_RANDOMIZATION COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_BUCKETS_SLICER
#define COLLECT_STATISTICS_BUCKETS_SLICER COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_BUCKETS_OVERFLOW_MAX_SLICER
#define COLLECT_STATISTICS_BUCKETS_OVERFLOW_MAX_SLICER COLLECT_STATISTICS_SLICER
#endif

#ifndef COLLECT_STATISTICS_ITERCOUNT_SLICER
#define COLLECT_STATISTICS_ITERCOUNT_SLICER COLLECT_STATISTICS_SLICER
#endif


#ifndef COLLECT_STATISTICS_RANDOMIZE_TRIALNUM_SLICER
#define COLLECT_STATISTICS_RANDOMIZE_TRIALNUM_SLICER COLLECT_STATISTICS_SLICER
#endif

// Note that ENABLE_IF_STATS_FOO(s) is defined as s and not s; The semicolon has to go inside the argument.
#if COLLECT_STATISTICS_XORPOPCNT_SLICER
    #define ENABLE_IF_STATS_XORPOPCNT_SLICER(s) s
#else
    #define ENABLE_IF_XORPOPCNT_SLICER(s)
#endif

#if COLLECT_STATISTICS_XORPOPCNT_PASS_SLICER
    #define ENABLE_IF_STATS_XORPOPCNT_PASS_SLICER(s) s
#else
    #define ENABLE_IF_STATS_XORPOPCNT_PASS_SLICER(s)
#endif

#if COLLECT_STATISTICS_FULLSCPRODS_SLICER
    #define ENABLE_IF_STATS_FULLSCPRODS_SLICER(s) s
#else
    #define ENABLE_IF_STATS_FULLSCPRODS_SLICER(s)
#endif

#if COLLECT_STATISTICS_REDSUCCESS_SLICER
    #define ENABLE_IF_STATS_REDSUCCESS_SLICER(s) s
#else
    #define ENABLE_IF_STATS_REDSUCCESS_SLICER(s)
#endif

#if COLLECT_STATISTICS_REPLACEMENTS_SLICER
    #define ENABLE_IF_STATS_REPLACEMENTS_SLICER(s) s
#else
    #define ENABLE_IF_STATS_REPLACEMENTS_SLICER(s)
#endif

#if COLLECT_STATISTICS_REPLACEMENTFAILURE_SLICER
    #define ENABLE_IF_STATS_REPLACEMENTFAILURE_SLICER(s) s
#else
    #define ENABLE_IF_STATS_REPLACEMENTFAILURE_SLICER(s)
#endif

#if COLLECT_STATISTICS_COLLISIONS_SLICER
    #define ENABLE_IF_STATS_COLLISIONS_SLICER(s) s
#else
    #define ENABLE_IF_STATS_COLLISIONS_SLICER(s)
#endif

#if COLLECT_STATISTICS_REDS_DURING_RANDOMIZATION
    #define ENABLE_IF_STATS_REDS_DURING_RANDOMIZATION(s) s
#else
    #define ENABLE_IF_STATS_REDS_DURING_RANDOMIZATION(s)
#endif

#if COLLECT_STATISTICS_BUCKETS_SLICER
    #define ENABLE_IF_STATS_BUCKETS_SLICER(s) s
#else
    #define ENABLE_IF_STATS_BUCKETS_SLICER(s)
#endif

#if COLLECT_STATISTICS_BUCKETS_OVERFLOW_MAX_SLICER
    #define ENABLE_IF_STATS_BUCKETS_OVERFLOW_MAX_SLICER(s) s
#else
    #define ENABLE_IF_STATS_BUCKETS_OVERFLOW_MAX_SLICER(s)
#endif

#if COLLECT_STATISTICS_BUCKETS_OVERFLOW_COUNT_SLICER
    #define ENABLE_IF_STATS_BUCKETS_OVERFLOW_COUNT_SLICER(s) s
#else
    #define ENABLE_IF_STATS_BUCKETS_OVERFLOW_COUNT_SLICER(s)
#endif

#if COLLECT_STATISTICS_ITERCOUNT_SLICER
    #define ENABLE_IF_STATS_ITERCOUNT_SLICER(s) s
#else
    #define ENABLE_IF_STATS_ITERCOUNT_SLICER(s)
#endif

#if COLLECT_STATISTICS_RANDOMIZE_TRIALNUM_SLICER
    #define ENABLE_IF_STATS_RANDOMIZE_TRIALNUM_SLICER(s) s
#else
    #define ENABLE_IF_STATS_RANDOMIZE_TRIALNUM_SLICER(s)
#endif

/**
    Actual SlicerStatistics class.
    It holds the statistics information as private data.
    We have public
        - getter functions get_stats_* to retrieve the data
        - static constexpr bool collect_statistics_* that tell whether the data is meaningful
            (This essentially allows to query what options we compiled with)
        - incrementer functions inc_stats_* to increment data.
        In some cases, we also have dec_stats_* to decrement data 
        and appropriate setters (eg. for BUCKETS_OVERFLOW_MAX_SLICER and ITERCOUNT_SLICER).
**/

class SlicerStatistics
{
private:

#if COLLECT_STATISTICS_XORPOPCNT_SLICER
    // std::atomic_ulong   stats_xorpopcnt;
    std::atomic_ulong   stats_xorpopcnt_s;
    std::atomic_ulong   stats_xorpopcnt_r;
#else 
    // static constexpr unsigned long stats_xorpopcnt = 0;
    static constexpr unsigned long stats_xorpopcnt_s = 0;
    static constexpr unsigned long stats_xorpopcnt_r = 0;
#endif

#if COLLECT_STATISTICS_XORPOPCNT_PASS_SLICER
    std::atomic_ulong   stats_xorpopcnt_pass_r;
    std::atomic_ulong   stats_xorpopcnt_pass_s;
#else 
    // static constexpr unsigned long stats_xorpopcnt_pass = 0;
    static constexpr unsigned long stats_xorpopcnt_pass_r = 0;
    static constexpr unsigned long stats_xorpopcnt_pass_s = 0;
#endif

#if COLLECT_STATISTICS_FULLSCPRODS_SLICER
    std::atomic_ulong   stats_fullscprods_r;
    std::atomic_ulong   stats_fullscprods_s;
#else 
    // static constexpr unsigned long stats_fullscprods = 0;
    static constexpr unsigned long stats_fullscprods_r = 0;
    static constexpr unsigned long stats_fullscprods_s = 0;
#endif

#if COLLECT_STATISTICS_REDSUCCESS_SLICER
    std::atomic_ulong   stats_redsucc_r;
    std::atomic_ulong   stats_redsucc_s;
#else 
    // static constexpr unsigned long stats_redsucc = 0;
    static constexpr unsigned long stats_redsucc_r = 0;
    static constexpr unsigned long stats_redsucc_s = 0;
#endif

#if COLLECT_STATISTICS_REPLACEMENTS_SLICER
    std::atomic_ulong   stats_replacements;
#else 
    static constexpr unsigned long stats_replacements = 0;
#endif

#if COLLECT_STATISTICS_COLLISIONS_SLICER
    std::atomic_ulong   stats_collisions_r;
    std::atomic_ulong   stats_collisions_s;
#else 
    static constexpr unsigned long stats_collisions_r = 0;
    static constexpr unsigned long stats_collisions_s = 0;
#endif

#if COLLECT_STATISTICS_REDS_DURING_RANDOMIZATION
    std::atomic_ulong   stats_reds_during_randomization;
#else 
    static constexpr unsigned long stats_reds_during_randomization = 0;
#endif

// #if COLLECT_STATISTICS_BUCKETS_SLICER
//     std::atomic_ulong   stats_bucknum;
// #else 
//     static constexpr unsigned long stats_bucknum = 0;
// #endif

#if COLLECT_STATISTICS_BUCKETS_OVERFLOW_MAX_SLICER
    std::atomic_ulong   stats_buck_over_max;
#else 
    static constexpr unsigned long stats_buck_over_max = 0;
#endif

#if COLLECT_STATISTICS_ITERCOUNT_SLICER
    std::atomic_ulong   stats_itercount_slicer;
#else 
    static constexpr unsigned long stats_itercount_slicer = 0;
#endif

#if COLLECT_STATISTICS_RANDOMIZE_TRIALNUM_SLICER
    std::atomic_ulong   stats_buck_over_num;
#else 
    static constexpr unsigned long stats_buck_over_num = 0;
#endif


/**
    To avoid at least some boilerplate, we use macros to create incrementers / getters:
    MAKE_ATOMIC_INCREMENTER(INCNAME,STAT) will create a
    function inc_stats_INCNAME(how_much) that increments stats_STAT by how_much (default:1)
    MAKE_ATOMIC_GETTER(GETTERNAME, STAT) will create a
    getter function get_stats_GETTERNAME() that returns stats_STAT
    MAKE_NOOP_INCREMENTER(INCNAME) create an incrementer that does nothing

    MAKE_INCREMENTER_FOR(INCNAME, STAT, NONTRIVIAL) will alias to MAKE_ATOMIC_GETTER if NONTRIVIAL is small positive preprocessor constant
    and alias MAKE_NOOP_INCREMENTER if NONTRIVIAL is the preprocessor constant 0.
    MAKE_INCREMENTER(NAME,NONTRIVIAL) is the same, but with INCNAME == NAME == STAT
    MAKE_GETTER_FOR(GETTERNAME, STAT, NONTRIVIAL) will create a getter function or a trivial getter function, depending on NONTRIVIAL
    MAKE_GETTER(GETTERNAME, NONTRIVIAL) will do the same, with STAT == GETTERNAME
    MAKE_GETTER_AND_INCREMENTER(NAME, NONTRIVIAL) will create both getter and incrementer, depending on NONTRIVIAL.
*/

#define MAKE_ATOMIC_INCREMENTER(INCNAME, STAT) \
void inc_stats_ ## INCNAME( mystd::decay_t<decltype(stats_##STAT.load())> how_much = 1) noexcept { stats_##STAT.fetch_add(how_much, std::memory_order_relaxed); }

#define MAKE_ATOMIC_DECREMENTER(INCNAME, STAT) \
void dec_stats_ ## INCNAME( mystd::decay_t<decltype(stats_##STAT.load())> how_much = 1) noexcept { stats_##STAT.fetch_sub(how_much, std::memory_order_relaxed); }

#define MAKE_ATOMIC_GETTER(GETTERNAME, STAT) \
FORCE_INLINE mystd::decay_t<decltype(stats_##STAT.load())> get_stats_##GETTERNAME() const noexcept { return stats_##STAT.load(); }

#define MAKE_ATOMIC_SETTER(SETTERNAME, STAT) \
void set_stats_##SETTERNAME(mystd::decay_t<decltype(stats_##STAT.load())> const new_val) noexcept {stats_##STAT.store(new_val); }

#define MAKE_NOOP_INCREMENTER(INCNAME) \
template<class Arg> FORCE_INLINE static void inc_stats_##INCNAME(Arg) noexcept {} \
FORCE_INLINE static void inc_stats_##INCNAME() noexcept {}

#define MAKE_NOOP_DECREMENTER(INCNAME) \
template<class Arg> FORCE_INLINE static void dec_stats_##INCNAME(Arg) noexcept {} \
FORCE_INLINE static void dec_stats_##INCNAME() noexcept {}

#define MAKE_NOOP_SETTER(SETTERNAME) \
template<class Arg> FORCE_INLINE static void set_stats_##SETTERNAME(Arg) noexcept {} \
FORCE_INLINE static void set_stats_##SETTERNAME() noexcept {}


/** Totally evil hackery to work around lack of C++ constexpr if (or #if's inside macro definitions...)
    Some gcc version might actually allow #if's inside macros, but we prefer portability. **/

#define MAKE_INCREMENTER_FOR(INCNAME, STAT, NONTRIVIAL) MAKE_INCREMENTER_AUX(INCNAME, STAT, NONTRIVIAL) // to macro-expand the name "NONTRIVIAL", such that token-pasting in the following macro is done AFTER macro expansion.
#define MAKE_INCREMENTER(INCNAME, NONTRIVIAL) MAKE_INCREMENTER_AUX(INCNAME, INCNAME, NONTRIVIAL)
#define MAKE_INCREMENTER_AUX(INCNAME, STAT, NONTRIVIAL) MAKE_INCREMENTER_##NONTRIVIAL(INCNAME, STAT) // This is evil
#define MAKE_INCREMENTER_0(INCNAME, STAT) MAKE_NOOP_INCREMENTER(INCNAME)
#define MAKE_INCREMENTER_1(INCNAME, STAT) MAKE_ATOMIC_INCREMENTER(INCNAME, STAT)
#define MAKE_INCREMENTER_2(INCNAME, STAT) MAKE_ATOMIC_INCREMENTER(INCNAME, STAT)
#define MAKE_INCREMENTER_3(INCNAME, STAT) MAKE_ATOMIC_INCREMENTER(INCNAME, STAT)
#define MAKE_INCREMENTER_4(INCNAME, STAT) MAKE_ATOMIC_INCREMENTER(INCNAME, STAT)

#define MAKE_DECREMENTER(NAME, NONTRIVIAL) MAKE_DECREMENTER_AUX(NAME, NAME, NONTRIVIAL)
#define MAKE_DECREMENTER_AUX(NAME, STAT, NONTRIVIAL) MAKE_DECREMENTER_##NONTRIVIAL(NAME, STAT)
#define MAKE_DECREMENTER_0(NAME, STAT) MAKE_NOOP_DECREMENTER(NAME)
#define MAKE_DECREMENTER_1(NAME, STAT) MAKE_ATOMIC_DECREMENTER(NAME, STAT)
#define MAKE_DECREMENTER_2(NAME, STAT) MAKE_ATOMIC_DECREMENTER(NAME, STAT)
#define MAKE_DECREMENTER_3(NAME, STAT) MAKE_ATOMIC_DECREMENTER(NAME, STAT)
#define MAKE_DECREMENTER_4(NAME, STAT) MAKE_ATOMIC_DECREMENTER(NAME, STAT)

#define MAKE_GETTER_FOR(GETTERNAME, STAT, NONTRIVIAL) MAKE_GETTER_AUX(GETTERNAME, STAT, NONTRIVIAL)
#define MAKE_GETTER(GETTERNAME, NONTRIVIAL) MAKE_GETTER_AUX(GETTERNAME, GETTERNAME, NONTRIVIAL)
#define MAKE_GETTER_AUX(GETTERNAME, STAT, NONTRIVIAL) MAKE_GETTER_##NONTRIVIAL(GETTERNAME, STAT)
#define MAKE_GETTER_0(GETTERNAME, STAT) \
FORCE_INLINE static constexpr auto get_stats_##GETTERNAME() noexcept -> mystd::remove_cv_t<decltype(stats_##STAT)> { return stats_##STAT; }
#define MAKE_GETTER_1(GETTERNAME, STAT) MAKE_ATOMIC_GETTER(GETTERNAME, STAT)
#define MAKE_GETTER_2(GETTERNAME, STAT) MAKE_ATOMIC_GETTER(GETTERNAME, STAT)
#define MAKE_GETTER_3(GETTERNAME, STAT) MAKE_ATOMIC_GETTER(GETTERNAME, STAT)
#define MAKE_GETTER_4(GETTERNAME, STAT) MAKE_ATOMIC_GETTER(GETTERNAME, STAT)

#define MAKE_SETTER_FOR(SETTERNAME, STAT, NONTRIVIAL) MAKE_SETTER_AUX(SETTERNAME, STAT, NONTRIVIAL)
#define MAKE_SETTER(SETTERNAME, NONTRIVIAL) MAKE_SETTER_AUX(SETTERNAME, SETTERNAME, NONTRIVIAL)
#define MAKE_SETTER_AUX(SETTERNAME, STAT, NONTRIVIAL) MAKE_SETTER_##NONTRIVIAL(SETTERNAME, STAT)
#define MAKE_SETTER_0(SETTERNAME, STAT) MAKE_NOOP_SETTER(SETTERNAME)
#define MAKE_SETTER_1(SETTERNAME, STAT) MAKE_ATOMIC_SETTER(SETTERNAME, STAT)
#define MAKE_SETTER_2(SETTERNAME, STAT) MAKE_ATOMIC_SETTER(SETTERNAME, STAT)
#define MAKE_SETTER_3(SETTERNAME, STAT) MAKE_ATOMIC_SETTER(SETTERNAME, STAT)
#define MAKE_SETTER_4(SETTERNAME, STAT) MAKE_ATOMIC_SETTER(SETTERNAME, STAT)

#define MAKE_GETTER_AND_INCREMENTER(NAME, NONTRIVIAL) \
MAKE_INCREMENTER(NAME, NONTRIVIAL) \
MAKE_GETTER(NAME, NONTRIVIAL)

#define MAKE_GETTER_SETTER_AND_INCREMENTER(NAME, NONTRIVIAL) \
MAKE_INCREMENTER(NAME, NONTRIVIAL) \
MAKE_GETTER(NAME, NONTRIVIAL) \
MAKE_SETTER(NAME, NONTRIVIAL)

public:
    static constexpr int collect_statistics_level = COLLECT_STATISTICS_SLICER;

    static constexpr bool collect_statistics_xorpopcnt  = (COLLECT_STATISTICS_XORPOPCNT_SLICER >= 1);
    MAKE_GETTER_AND_INCREMENTER(xorpopcnt_r, COLLECT_STATISTICS_XORPOPCNT_SLICER)
    MAKE_GETTER_AND_INCREMENTER(xorpopcnt_s, COLLECT_STATISTICS_XORPOPCNT_SLICER)
    unsigned long get_stats_xorpopcnt_total() const { return get_stats_xorpopcnt_s() + get_stats_xorpopcnt_r(); }

    static constexpr bool collect_statistics_xorpopcnt_pass  = (COLLECT_STATISTICS_XORPOPCNT_PASS_SLICER >= 1);
    MAKE_GETTER_AND_INCREMENTER(xorpopcnt_pass_r, COLLECT_STATISTICS_XORPOPCNT_PASS_SLICER)
    MAKE_GETTER_AND_INCREMENTER(xorpopcnt_pass_s, COLLECT_STATISTICS_XORPOPCNT_PASS_SLICER)
    unsigned long get_stats_xorpopcnt_pass_total() const { return get_stats_xorpopcnt_pass_s() + get_stats_xorpopcnt_pass_r(); }

    static constexpr bool collect_statistics_fullscprods  = (COLLECT_STATISTICS_FULLSCPRODS_SLICER >= 1);
    MAKE_GETTER_AND_INCREMENTER(fullscprods_r, COLLECT_STATISTICS_FULLSCPRODS_SLICER)
    MAKE_GETTER_AND_INCREMENTER(fullscprods_s, COLLECT_STATISTICS_FULLSCPRODS_SLICER)
    unsigned long get_stats_fullscprods_total() const { return get_stats_fullscprods_s() + get_stats_fullscprods_r(); }

    // static constexpr bool collect_statistics_filterpass;

    static constexpr bool collect_statistics_redsucc  = (COLLECT_STATISTICS_REDSUCCESS_SLICER >= 1);
    MAKE_GETTER_AND_INCREMENTER(redsucc_r, COLLECT_STATISTICS_REDSUCCESS_SLICER)
    MAKE_GETTER_AND_INCREMENTER(redsucc_s, COLLECT_STATISTICS_REDSUCCESS_SLICER)
    unsigned long get_stats_redsucc_total() const { return get_stats_redsucc_s() + get_stats_redsucc_r(); }

    static constexpr bool collect_statistics_replacements  = (COLLECT_STATISTICS_REPLACEMENTS_SLICER >= 1);
    MAKE_GETTER_AND_INCREMENTER(replacements, COLLECT_STATISTICS_REPLACEMENTS_SLICER)

    static constexpr bool collect_statistics_collisions  = (COLLECT_STATISTICS_COLLISIONS_SLICER >= 1);
    MAKE_GETTER_AND_INCREMENTER(collisions_r, COLLECT_STATISTICS_COLLISIONS_SLICER)
    MAKE_GETTER_AND_INCREMENTER(collisions_s, COLLECT_STATISTICS_COLLISIONS_SLICER)
    unsigned long get_stats_collisions() const { return get_stats_collisions_s() + get_stats_collisions_r(); }

    static constexpr bool collect_statistics_reds_during_randomization  = (COLLECT_STATISTICS_REDS_DURING_RANDOMIZATION >= 1);
    MAKE_GETTER_AND_INCREMENTER(reds_during_randomization, COLLECT_STATISTICS_REDS_DURING_RANDOMIZATION)

    // static constexpr bool collect_statistics_bucknum  = (COLLECT_STATISTICS_BUCKETS_SLICER >= 1);
    // MAKE_GETTER_AND_INCREMENTER(bucknum, COLLECT_STATISTICS_BUCKETS_SLICER)

    static constexpr bool collect_statistics_buck_over_max  = (COLLECT_STATISTICS_BUCKETS_OVERFLOW_MAX_SLICER >= 1);
    MAKE_GETTER_SETTER_AND_INCREMENTER(buck_over_max, COLLECT_STATISTICS_BUCKETS_OVERFLOW_MAX_SLICER)

    static constexpr bool collect_statistics_buck_over_num  = (COLLECT_STATISTICS_RANDOMIZE_TRIALNUM_SLICER >= 1);
    MAKE_GETTER_AND_INCREMENTER(buck_over_num, COLLECT_STATISTICS_RANDOMIZE_TRIALNUM_SLICER)

    static constexpr bool collect_statistics_itercount_slicer = (COLLECT_STATISTICS_ITERCOUNT_SLICER >= 1);
    MAKE_GETTER_SETTER_AND_INCREMENTER(itercount_slicer, COLLECT_STATISTICS_ITERCOUNT_SLICER)

    inline void clear_statistics() noexcept
    {
    #if COLLECT_STATISTICS_XORPOPCNT_SLICER
        stats_xorpopcnt_r = 0;
        stats_xorpopcnt_s = 0;
    #endif

    #if COLLECT_STATISTICS_XORPOPCNT_PASS_SLICER
        stats_xorpopcnt_pass_r = 0;
        stats_xorpopcnt_pass_s = 0;
    #endif

    #if COLLECT_STATISTICS_FULLSCPRODS_SLICER
        stats_fullscprods_r = 0;
        stats_fullscprods_s = 0;
    #endif

    #if COLLECT_STATISTICS_REDSUCCESS_SLICER
        stats_redsucc_r = 0;
        stats_redsucc_s = 0;
    #endif

    #if COLLECT_STATISTICS_REPLACEMENTS_SLICER
        stats_replacements = 0;
    #endif

    #if COLLECT_STATISTICS_COLLISIONS_SLICER
        stats_collisions_r = 0;
        stats_collisions_s = 0;
    #endif

    #if COLLECT_STATISTICS_REDS_DURING_RANDOMIZATION
        stats_reds_during_randomization = 0;
    #endif

    // #if COLLECT_STATISTICS_BUCKETS_SLICER
    //     stats_bucknum = 0;
    // #endif

    #if COLLECT_STATISTICS_BUCKETS_OVERFLOW_MAX_SLICER
        stats_buck_over_max = 0;
    #endif

    #if COLLECT_STATISTICS_ITERCOUNT_SLICER
        stats_itercount_slicer = 0;
    #endif

    #if COLLECT_STATISTICS_RANDOMIZE_TRIALNUM_SLICER
        stats_buck_over_num = 0;
    #endif
    }

    void print_statistics(std::ostream &os = std::cout)
    {
                //#ifdef COLLECT_STATISTICS_SLICER
                //std::cout << " - - - <STATISTIC> - - -" << std::endl;
                //#endif

                if(collect_statistics_xorpopcnt)
                {
                    os << "XORpopcnt calls: " << get_stats_xorpopcnt_total() << std::endl;
                    os << "while randomizing: " << get_stats_xorpopcnt_r();
                    os << "\n";
                    os << "while slicing: " << get_stats_xorpopcnt_s();
                    os << "\n";
                }
                if(collect_statistics_xorpopcnt_pass)
                {
                    os << "XORpopcnt passes: " << get_stats_xorpopcnt_pass_total() << std::endl;
                    os << "while randomizing: " << get_stats_xorpopcnt_pass_r();
                    os << "\n";
                    os << "while slicing: " << get_stats_xorpopcnt_pass_s();
                    os << "\n";
                }
                if(collect_statistics_fullscprods)
                {
                    os << "Total scalar prods: " << get_stats_fullscprods_total();
                    os << "\n";
                }
                if(collect_statistics_redsucc)
                {
                    os << "Succ. reductions: " << get_stats_redsucc_total();
                    os << "\n";
                }
                if(collect_statistics_replacements)
                {
                    os << "dbt replacements: " << get_stats_replacements();
                    os << "\n";
                }

                if(collect_statistics_collisions)
                {
                    os << "dbt collisions: " << get_stats_collisions() << std::endl;
                    os << "while randomizing: " << get_stats_collisions_r();
                    os << "\n";
                    os << "while slicing: " << get_stats_collisions_s();
                    os << "\n";
                }

                if(collect_statistics_reds_during_randomization)
                {
                    os << "dbt reds_during_randomization: " << get_stats_reds_during_randomization();
                    os << "\n";
                }
                // if(collect_statistics_bucknum)
                // {
                //     os << "dbt bucknum: " << get_stats_bucknum();
                //     os << "\n";
                // }
                if(collect_statistics_buck_over_max)
                {
                    os << "dbt buck_over_max: " << get_stats_buck_over_max();
                    os << "\n";
                }
                if(collect_statistics_buck_over_num)
                {
                    os << "dbt buck_over_num: " << get_stats_buck_over_num();
                    os << "\n";
                }
                if(collect_statistics_itercount_slicer){
                    os << "iterations: " << get_stats_itercount_slicer();
                    os << "\n";
                }
                //#ifdef COLLECT_STATISTICS_SLICER
                //std::cout << " - - - <END STATISTIC> - - -" << std::endl;
                //#endif
    }

};
