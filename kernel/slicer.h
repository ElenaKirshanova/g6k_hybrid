//
// Created by Elena Kirshanova on 07/09/2024.
//


#ifndef G6K_HYBRID_SLICER_H
#define G6K_HYBRID_SLICER_H
#endif

static constexpr unsigned int XPC_SLICER_SAMPLING_THRESHOLD = 75; // XPC Threshold for iterative slicer sampling //75
static constexpr unsigned int XPC_SLICER_THRESHOLD = 96; // XPC Threshold for iterative slicer sampling //96

#define REDUCE_DIST_MARGIN 1.008
#define REDUCE_DIST_MARGIN_HALF 1.004

#ifndef MAX_SIEVING_DIM
#define MAX_SIEVING_DIM 128
#endif

#include "compat.hpp"
#include "statistics_slicer.hpp"


struct Entry_t
{
    std::array<LFT,MAX_SIEVING_DIM> yr;     // Vector coordinates in gso basis renormalized by the rr[i] (for faster inner product)
    CompressedVector c;                     // Compressed vector (i.e. a simhash)
    UidType uid;                            // Unique identifier for collision detection (essentially a hash)
    FT len = 0.;                            // (squared) length of the vector, renormalized by the local gaussian heuristic
    IT i;                                   // Index in Unique_entry_t
    //std::array<LFT,OTF_LIFT_HELPER_DIM> otf_helper; // auxiliary information to accelerate otf lifting of pairs, commented out for slicer
};

struct Unique_entry_t
{
    std::array<LFT,MAX_SIEVING_DIM> yr_o;   // Vector coos in gso basis for the input (non-randomized) target; needed for applications of the slicer (hybrid)
};


struct QEntry;
class ProductLSH;

class RandomizedSlicer{

public:
    explicit RandomizedSlicer(Siever &sieve, unsigned long int seed = 0) :
            sieve(sieve), db_t(), cdb_t(), n(0), rng_t(seed), sim_hashes_t(rng_t.rng_nolock())
    {
        this->n = this->sieve.n;
        sim_hashes_t.reset_compress_pos(this->sieve);
        uid_hash_table_t.reset_hash_function(this->sieve);
        this->statistics.clear_statistics();
    }

    friend SimHashes;
    friend UidHashTable;

    enum class RecomputeSlicer // used as a bitmask for the template argument to recompute_data_for_entry below
    {
        none = 0,
        recompute_yr = 1,
        recompute_len = 2,
        recompute_c = 4,
        recompute_uid = 8,
        recompute_otf_helper = 16,
        recompute_all = 31,
        consider_otf_lift = 32,
        recompute_all_and_consider_otf_lift = 63
    };

    Siever &sieve;

    CACHELINE_VARIABLE(std::vector<Entry_t>, db_t);             // database of targets
    CACHELINE_VARIABLE(std::vector<CompressedEntry>, cdb_t);  // compressed version, faster access and periodically sorted
    CACHELINE_VARIABLE(std::vector<CompressedEntry>, cdb_t_tmp_copy); // for sorting
    CACHELINE_VARIABLE(std::vector<Unique_entry_t>, unique_db);  //to store unique targets
    CACHELINE_VARIABLE(rng::threadsafe_rng, rng_t);

    // collects various statistics about the slicer. Details about statistics collection are in statistics_slicer.hpp
    CACHELINE_VARIABLE(SlicerStatistics, statistics);

    unsigned int n;

    unsigned int Nt = 1;  //number of unique targets
    FT saturation_scalar = 1.1; // Nt*saturation_scalar = number of vectors of length < proj_error_bound required to terminate
    FT proj_error_bound = 0.9; //arbitrary value, expect to be set by the caller

    size_t MAX_SLICER_ITERS = 1000;

    SimHashes sim_hashes_t; // needs to go after rng!
    UidHashTable uid_hash_table_t; //hash table for db_t -- the database of targets

    size_t threads = 1;

    thread_pool::thread_pool threadpool;
    size_t sorted_until = 0;

    const char* filename_cdbt = "cdbt_out.txt";

    void parallel_sort_cdb();


    void randomize_target_small_task(Entry_t &t);
    void grow_db_with_target(const double t_yr[], size_t n_per_target);

    bool bdgl_like_sieve(size_t nr_buckets_aim, const size_t blocks, const size_t multi_hash, bool verbose, bool showstats);
    void slicer_bucketing(const size_t blocks, const size_t multi_hash, const size_t nr_buckets_aim,
                                            std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index);
    void slicer_bucketing_task(const size_t t_id, std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index, ProductLSH &lsh);

    void slicer_process_buckets(const std::vector<uint32_t> &buckets, const std::vector<atomic_size_t_wrapper> &buckets_index,
                                std::vector<std::vector<QEntry>> &t_queues);
    void slicer_process_buckets_task(const size_t t_id, const std::vector<uint32_t> &buckets,
                                     const std::vector<atomic_size_t_wrapper> &buckets_index, std::vector<QEntry> &t_queue);
    std::pair<LFT, int8_t> reduce_to_QEntry_t(CompressedEntry *ce1, CompressedEntry *ce2);

    void slicer_queue_dup_remove_task( std::vector<QEntry> &queue);
    void slicer_queue(std::vector<std::vector<QEntry>> &t_queues, std::vector<std::vector<Entry_t>>& transaction_db );
    void slicer_queue_create_task( const std::vector<QEntry> &queue, std::vector<Entry_t> &transaction_db, int64_t &write_index);
    inline int slicer_reduce_with_delayed_replace(const size_t i1, const size_t i2, std::vector<Entry_t>& transaction_db, int64_t& write_index, LFT new_l, int8_t sign);
    size_t slicer_queue_insert_task( const size_t t_id, std::vector<Entry_t> &transaction_db, int64_t write_index);
    bool slicer_replace_in_db(size_t cdb_index, Entry_t &e);

    void set_nthreads(size_t nt){ this->threads = nt;}
    void set_proj_error_bound(FT len) {this->proj_error_bound = len;}
    void set_max_slicer_interations(size_t maxiter){this->MAX_SLICER_ITERS = maxiter;}
    void set_Nt(unsigned int nt) {this->Nt = nt;}
    void set_saturation_scalar(FT sat_scalar) {this->saturation_scalar = sat_scalar;}
    void set_filename_cdbt(const char* filename_prefix) {this->filename_cdbt = filename_prefix;}

    bool dump_cdb_t(const char* filename_prefix, size_t it);


    template<RecomputeSlicer what_to_recompute>
    inline void recompute_data_for_entry_t(Entry_t &e);
};
