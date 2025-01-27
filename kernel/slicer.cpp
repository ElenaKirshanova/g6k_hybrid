#include "siever.h"
#include "slicer.h"
#include "fht_lsh.h"
#include <chrono>

#include "time_measurement.h"

void update_curtime_slicer( double &cur_time, std::chrono::time_point<std::chrono::high_resolution_clock> start, std::chrono::time_point<std::chrono::high_resolution_clock> finish ){
            long long delta_t;
            double delta_t_dbl;
            delta_t = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
            delta_t_dbl = (double)delta_t / 1000000000.0;
            cur_time += delta_t_dbl;
}

inline bool compare_QEntry(QEntry const& lhs, QEntry const& rhs) { return lhs.len > rhs.len; }

ENABLE_BITOPS_FOR_ENUM(RandomizedSlicer::RecomputeSlicer)

template<RandomizedSlicer::RecomputeSlicer what_to_recompute>
inline void RandomizedSlicer::recompute_data_for_entry_t(Entry_t &e)
{
    //ATOMIC_CPUCOUNT(214);
    bool constexpr rec_len = (what_to_recompute & RecomputeSlicer::recompute_len) != RecomputeSlicer::none;
    bool constexpr rec_c = (what_to_recompute & RecomputeSlicer::recompute_c) != RecomputeSlicer::none;
    bool constexpr rec_uid = (what_to_recompute & RecomputeSlicer::recompute_uid) != RecomputeSlicer::none;


    CPP17CONSTEXPRIF(rec_len) e.len = 0.;
    if (rec_len)
    {
        for (unsigned int i = 0; i < n; ++i)
        {
            e.yr[i] = static_cast<FT>(e.yr[i]);
            e.len+=e.yr[i] * e.yr[i]; // slightly inefficient if we only compute the length and not yr, but that does not happen anyway.
        }
    }

    CPP17CONSTEXPRIF (rec_uid)
    {
        e.uid  = uid_hash_table_t.compute_uid_t(e.yr);
    }

    CPP17CONSTEXPRIF (rec_c)
    {
        e.c = this->sieve.sim_hashes.compress(e.yr);
    }

    return;
}

//First element is from the list of targets, the second is from the siever db
std::pair<LFT, int8_t> RandomizedSlicer::reduce_to_QEntry_t(CompressedEntry *ce1, CompressedEntry *ce2)
{
    LFT inner = std::inner_product(db_t[ce1->i].yr.begin(), db_t[ce1->i].yr.begin()+n, this->sieve.db[ce2->i].yr.begin(),  static_cast<LFT>(0.));
    LFT new_l = ce1->len + ce2->len - 2 * std::abs(inner);
    int8_t sign = (inner < 0 ) ? 1 : -1;
    return { new_l, sign };
}

void RandomizedSlicer::parallel_sort_cdb() {

    assert(sorted_until <= cdb_t.size());
    assert(std::is_sorted(cdb_t.cbegin(), cdb_t.cbegin() + sorted_until, compare_CE()));
    if (sorted_until == cdb_t.size()) {
        return;
    }
    pa::sort(cdb_t.begin() + sorted_until, cdb_t.end(), compare_CE(), threadpool);
    cdb_t_tmp_copy.resize(cdb_t.size());
    pa::merge(cdb_t.begin(), cdb_t.begin() + sorted_until, cdb_t.begin() + sorted_until, cdb_t.end(),
              cdb_t_tmp_copy.begin(), compare_CE(), threadpool);
    cdb_t.swap(cdb_t_tmp_copy);
    sorted_until = cdb_t.size();
    //for(unsigned int i = 0; i<cdb_t.size(); i++)
    //    std::cout << cdb_t[i].len << " ";
    //std::cout << std::endl;
    assert(std::is_sorted(cdb_t.cbegin(), cdb_t.cend(), compare_CE()));
    return;
}


void RandomizedSlicer::randomize_target_small_task(Entry_t &t)
{
    size_t fullS = this->sieve.cdb.size();
    size_t max_trial = 2 + std::pow(fullS, .3);
    CompressedEntry* fast_cdb = &(this->sieve.cdb.front());
    size_t partial = 5 * std::pow(fullS, .6);


    Entry_t tmp;

    size_t i = (rng_t() % partial);
    size_t j = (rng_t() % partial);

    for (unsigned int trial=0;;++trial) {
        ++j;
        unsigned int w1 = 0;
        //unsigned int w2 = 0;


        for (size_t k = 0; k < XPC_WORD_LEN; ++k) {
            w1 += __builtin_popcountl(fast_cdb[i].c[k] ^ fast_cdb[j].c[k]);
        }
        //std::cout << "w1: " << w1 << std::endl;
        if (w1 < XPC_SLICER_SAMPLING_THRESHOLD || w1 > (XPC_BIT_LEN - XPC_SLICER_SAMPLING_THRESHOLD) || trial > max_trial) {
            if (i == j) continue;
            ZT sign = w1 < XPC_SLICER_SAMPLING_THRESHOLD / 2 ? -1 : 1;

            //std::cout << "good w1: " << w1 << std::endl;

            std::array<LFT,MAX_SIEVING_DIM> new_yr = this->sieve.db[this->sieve.cdb[i].i].yr;

            //for( size_t ii = 0; ii < n; ii++ ) {
            //    new_yr[ii] += sign*this->sieve.db[this->sieve.cdb[j].i].yr[ii];
            //}

            this->sieve.addsub_vec(new_yr, this->sieve.db[this->sieve.cdb[j].i].yr, static_cast<ZT>(sign));

            //TODO: change to XPC (if makes sense)
            LFT const inner = std::inner_product(t.yr.begin(), t.yr.begin() + n, new_yr.begin(), static_cast<LFT>(0.));
            sign = inner < 0 ? 1 : -1;
            this->sieve.addsub_vec(t.yr, new_yr, static_cast<ZT>(sign));

            recompute_data_for_entry_t<RandomizedSlicer::RecomputeSlicer::recompute_all>(t);
            break;
        }
    }

}

void RandomizedSlicer::grow_db_with_target(const double t_yr[], size_t n_per_target){
    //std::cout << "in grow_db"  << std::endl;
    Entry_t input_t;

    for(int i = 0; i < MAX_SIEVING_DIM; i++){
        input_t.yr[i] = t_yr[i];
    }

    recompute_data_for_entry_t<RandomizedSlicer::RecomputeSlicer::recompute_all>(input_t);


    //std::cout << "length: " << input_t.len << " uid: " <<input_t.uid << std::endl;

    unsigned long const start = db_t.size();
    unsigned long const N = start+n_per_target;

    if(!uid_hash_table_t.insert_uid(input_t.uid)){
        std::cerr << "The original target is already in db" << std::endl;
        // exit(0);
        return;
    }

    db_t.reserve(N);
    cdb_t.reserve(N);
    db_t.resize(N);
    cdb_t.resize(N);

    db_t[start] = input_t;
    CompressedEntry ce;
    ce.len = input_t.len;
    ce.c = input_t.c;
    ce.i = start;
    cdb_t[start] = ce;

    for( size_t i = start+1; i < N; i++)
    {
        int col = 0;

        Entry_t e = input_t;

        for ( col = 0; col < 10; ++col)
        {
            Entry_t tmp = e;
            randomize_target_small_task(tmp);

            if(!uid_hash_table_t.insert_uid(tmp.uid)) {
                continue;
            }
            db_t[i] = tmp;

            CompressedEntry ce;
            ce.len = tmp.len;
            ce.c = tmp.c;
            ce.i = i;
            cdb_t[i] = ce;

            if (tmp.len < 0.99*input_t.len) std::cout << "reduced the target during randomization" << std::endl;

            break;

        }
        if(col>=64)
        {
            std::cerr << "Error : All new randomizations collide." << std::endl;
            exit(1);
        }

    }

}

inline int RandomizedSlicer::slicer_reduce_with_delayed_replace(const size_t i1, const size_t i2,  std::vector<Entry_t>& transaction_db, int64_t& write_index, LFT new_l, int8_t sign)
{
    if (new_l < REDUCE_DIST_MARGIN*db_t[i1].len)
    {

        std::array<LFT,MAX_SIEVING_DIM> new_yr = db_t[i1].yr;
        this->sieve.addsub_vec(new_yr,  this->sieve.db[i2].yr, static_cast<ZT>(sign));
        UidType new_uid = uid_hash_table_t.compute_uid_t(new_yr);

        if( !uid_hash_table_t.check_uid_unsafe(new_uid) && uid_hash_table_t.insert_uid(new_uid) )
        {
            int64_t index = write_index--; // atomic and signed!
            if( index >= 0 ) {
                Entry_t& new_entry = transaction_db[index];
                new_entry.yr = new_yr;
                recompute_data_for_entry_t<RandomizedSlicer::RecomputeSlicer::recompute_all>(new_entry);

                return 1;
            }
            std::cout << "transaction_db full" << std::endl;
            return -2; // transaction_db full
        }
        else
        {
            return 0;
        }
    }
    return -1;
}


void RandomizedSlicer::slicer_queue_create_task(const std::vector<QEntry> &queue, std::vector<Entry_t> &transaction_db, int64_t &write_index) {
    const size_t Q = queue.size();

    for(unsigned int index = 0; index < Q; index++ )  {

        if( queue[index].sign == 0 ){
            continue;
        }

        slicer_reduce_with_delayed_replace( queue[index].i, queue[index].j,
                                          transaction_db, write_index, queue[index].len, queue[index].sign);
        if( write_index < 0 ){
            //std::cerr << "Spilling full transaction db " << t_id << " " << Q << " " << index << std::endl;
            //exit(1);
            break;
        }
    }
}

bool RandomizedSlicer::slicer_replace_in_db(size_t cdb_index, Entry_t &e)
{
    CompressedEntry &ce = cdb_t[cdb_index];

    if (REDUCE_DIST_MARGIN * e.len >= ce.len)
    {
        uid_hash_table_t.erase_uid(e.uid);
        return false;
    }
    uid_hash_table_t.erase_uid(db_t[ce.i].uid);
    ce.len = e.len;
    ce.c = e.c;
    db_t[ce.i] = e;
    return true;
}

size_t RandomizedSlicer::slicer_queue_insert_task( const size_t t_id, std::vector<Entry_t> &transaction_db, int64_t write_index) {
    const size_t S = cdb_t.size();
    const size_t insert_after = std::max(int(0), int(int(S)-1-t_id-threads*(transaction_db.size()-write_index)));
    size_t kk = S-1 - t_id;
    for( int i = transaction_db.size()-1; i > write_index and kk >= insert_after; --i )  {
        if( slicer_replace_in_db( kk, transaction_db[i] ) ) {
            kk -= threads;
        }
    }
    return kk + threads;
}

void RandomizedSlicer::slicer_queue(std::vector<std::vector<QEntry>> &t_queues, std::vector<std::vector<Entry_t>>& transaction_db ) {

    const size_t S = cdb_t.size();
    size_t Q = 0;

    for(unsigned int i = 0; i < threads; i++)
        Q += t_queues[i].size();

    size_t insert_after = std::max(0, int(int(S)-Q));

    for(unsigned int i = 0; i < threads; i++ )
        transaction_db[i].resize(std::min(S-insert_after, Q)/threads + 1);


    std::vector<int> write_indices(threads, transaction_db[0].size()-1);


    // Prepare transaction DB from queue
    for( size_t t_id = 0; t_id < threads; ++t_id ) {
        threadpool.push([this, t_id, &t_queues, &transaction_db,&write_indices](){
            int64_t write_index = write_indices[t_id];
            slicer_queue_create_task(t_queues[t_id], transaction_db[t_id], write_index);
            write_indices[t_id] = write_index;
            t_queues[t_id].clear();
        });
    }
    threadpool.wait_work();

    // Insert transaction DB
    std::vector<size_t> kk(threads);
    for( size_t t_id = 0; t_id < threads; ++t_id ) {
        threadpool.push([this, &kk, t_id, &transaction_db,&write_indices](){
            kk[t_id] = slicer_queue_insert_task(t_id, transaction_db[t_id], write_indices[t_id]);
        });
    }

    //std::cout << "slicer_queue_insert_task finished" << std::endl;

    threadpool.wait_work();
    size_t min_kk = kk[0];
    size_t inserted = 0;
    for(unsigned int i = 0; i < threads; i++ ){
        min_kk = std::min(min_kk, kk[i]);
        inserted += (S-1-i - kk[i]-threads)/threads;
    }
    sorted_until = min_kk;
}

// assumes buckets and buckets_index are resized and resetted correctly.
void RandomizedSlicer::slicer_bucketing(const size_t blocks, const size_t multi_hash, const size_t nr_buckets_aim,
                            std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index)
{
    // init hash
    //std::cout << "lsh_seed from sieve: " << this->sieve.lsh_seed << std::endl;
    //std::cout << "n = " << n << std::endl;
    ProductLSH lsh(n, blocks, nr_buckets_aim, multi_hash, this->sieve.lsh_seed);

    const size_t nr_buckets = lsh.codesize;
    const size_t S = cdb_t.size();
    size_t bsize = 4 * (S*multi_hash / double(nr_buckets)); //factor of 4 gives decent success probability for slicer
    buckets.resize( nr_buckets * bsize );
    buckets_index.resize(nr_buckets);
    for( size_t i = 0; i < nr_buckets; i++ )
        buckets_index[i].val = 0;

    for (size_t t_id = 0; t_id < threads; ++t_id)
    {
        threadpool.push([this, t_id, &buckets, &buckets_index, &lsh](){
            slicer_bucketing_task(t_id, buckets, buckets_index, lsh);
        });
    }
    threadpool.wait_work();

    for( size_t i = 0; i < nr_buckets; ++i ) {
        // bucket overflow
        if( buckets_index[i].val > bsize ) {
            buckets_index[i].val = bsize;
            //std::cout << "slicer: bucket overflow!" << std::endl;
        }
    }
    //exit(1);
}


void RandomizedSlicer::slicer_bucketing_task(const size_t t_id, std::vector<uint32_t> &buckets, std::vector<atomic_size_t_wrapper> &buckets_index, ProductLSH &lsh) {

    CompressedEntry* const fast_cdb = cdb_t.data();
    const size_t S = cdb_t.size();
    const size_t multi_hash = lsh.multi_hash;
    const unsigned int nr_buckets = buckets_index.size();
    const size_t bsize = buckets.size() / nr_buckets;
    //const size_t threads = threads;

    uint32_t i_start = t_id;
    int32_t res[multi_hash];
    size_t bucket_index = 0;
    for (uint32_t i = i_start; i < S; i += threads)
    {
        auto db_index = fast_cdb[i].i;
        //for (size_t ii =0; ii<n; ii++) std::cout << db_t[db_index].yr[ii] << " " ;
        //std::cout << std::endl;
        lsh.hash( db_t[db_index].yr.data() , res);
        for( size_t j = 0; j < multi_hash; j++ ) {
            uint32_t b = res[j];
            assert( b < nr_buckets );
            bucket_index = buckets_index[b].val++; // atomic
            if( bucket_index < bsize ) {
                buckets[bsize * b + bucket_index] = i;
            }
        }
    }

}

// Returned queue is sorted
void RandomizedSlicer::slicer_process_buckets(const std::vector<uint32_t> &buckets, const std::vector<atomic_size_t_wrapper> &buckets_index,
                                  std::vector<std::vector<QEntry>> &t_queues)
{
    for (size_t t_id = 0; t_id < threads; ++t_id)
    {
        threadpool.push([this, t_id, &buckets, &buckets_index, &t_queues](){slicer_process_buckets_task(t_id, buckets, buckets_index, t_queues[t_id]);});
    }
    threadpool.wait_work();
}


void RandomizedSlicer::slicer_process_buckets_task(const size_t t_id,
                                       const std::vector<uint32_t> &buckets,
                                       const std::vector<atomic_size_t_wrapper> &buckets_index, std::vector<QEntry> &t_queue)
{

    const size_t nr_buckets = buckets_index.size();
    const size_t bsize = buckets.size() / nr_buckets;

    const uint32_t* const fast_buckets_t = buckets.data();
    const uint32_t* const fast_buckets = this->sieve.buckets.data();

    CompressedEntry* const fast_cdb_t = cdb_t.data();
    CompressedEntry* const fast_cdb = this->sieve.cdb.data();

    const size_t S = cdb_t.size();

    int64_t kk = S-1-t_id; //controls the number of new reduced vectors to be added

    //LFT lenbound = fast_cdb[std::min(S-1, size_t(params.bdgl_improvement_db_ratio * S))].len;
    const size_t b_start = t_id;

    for (size_t b = b_start; b < nr_buckets; b += threads)
    {
        const size_t i_start = bsize * b;
        const size_t i_end = bsize * b + buckets_index[b].val;

        const size_t bsize_sieve = this->sieve.buckets.size() / this->sieve.buckets_i.size();
        const size_t i_start_s = bsize_sieve*b;
        const size_t i_end_s = i_start_s + this->sieve.buckets_i[b].val;


        //B +=( (i_end - i_start) * (i_end-i_start-1)) / 2;
        for( size_t i = i_start; i < i_end; ++i )
        {
            if (kk < .1 * S) break;

            uint32_t bi = fast_buckets_t[i];
            CompressedEntry *pce1 = &fast_cdb_t[bi];
            CompressedVector cv = pce1->c;

            LFT best_reduction = pce1->len;
            int best_j = -1;
            int best_sign = 0;

            for (size_t j = i_start_s; j < i_end_s; ++j)
            {
                uint32_t bj = fast_buckets[j];
                if( this->sieve.is_reducible_maybe<XPC_SLICER_THRESHOLD>(cv, fast_cdb[bj].c) ) //TODO:adjust XPC_SLICER_THRESHOLD
                {

                    std::pair<LFT, int> len_and_sign = reduce_to_QEntry_t( pce1, &fast_cdb[bj] );
                    //if( len_and_sign.first < 0.98*pce1->len)
                    if(len_and_sign.first < best_reduction)
                    {
                        best_j = j;
                        best_reduction = len_and_sign.first;
                        best_sign = len_and_sign.second;

                        if (kk < .1 * S) break;
                        kk -= threads;
                        //t_queue.push_back({ pce1->i, fast_cdb[bj].i, len_and_sign.first, (int8_t)len_and_sign.second});

                    }
                    //else if( params.otf_lift and len_and_sign.first < params.lift_radius ) {
                    //    bdgl_lift( pce1->i, fast_cdb[bj].i, len_and_sign.first, len_and_sign.second );
                    //}
                }
            }
            if(best_j!=-1) {
                t_queue.push_back({ pce1->i, fast_cdb[fast_buckets[best_j]].i, best_reduction, (int8_t)best_sign});
                this->sieve.siever_vect_access[fast_cdb[fast_buckets[best_j]].i] += 1;
            }
        }
    }
    std::sort( t_queue.begin(), t_queue.end(), &compare_QEntry);
}

void RandomizedSlicer::print_siever_vect_access(){
    size_t l = this->sieve.db_size();
    std::cout << "siever_vect_access = [";
    for (size_t tmp=0; tmp<l-1; tmp++){
        std::cout << this->sieve.siever_vect_access[tmp] << ", ";
    }
    std::cout << this->sieve.siever_vect_access[l-1] << "]";
}

bool RandomizedSlicer::bdgl_like_sieve(size_t nr_buckets_aim, const size_t blocks, const size_t multi_hash, bool verbose){

    //std::cout << "nr_buckets_aim:" << nr_buckets_aim << " blocks: " << blocks << " multi_hash: " <<multi_hash <<  std::endl;
    parallel_sort_cdb();

    std::vector<std::vector<Entry_t>> transaction_db(threads, std::vector<Entry_t>());
    std::vector<uint32_t> buckets;
    std::vector<atomic_size_t_wrapper> buckets_i;
    std::vector<std::vector<QEntry>> t_queues(threads);

    std::vector<uint32_t> vec(this->sieve.db_size(),0);
    this->sieve.siever_vect_access = vec;

    //TODO: assert that all input parameters are equal to those from bdgl_sieve

    size_t it = 0;
    // std::chrono::time_point<std::chrono::high_resolution_clock> start;
    // std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    // long long delta_t;
    // double delta_t_dbl;
    
        unsigned long long curit;
    hr_time start;
    hr_time start_bdgl_bucketing;
    hr_time start_slicer_bucketing;
    hr_time start_slicer_process_buckets;
    hr_time start_slicer_queue;
    hr_time start_parallel_sort_cdb;

    hr_time finish;
    hr_time finish_bdgl_bucketing;
    hr_time finish_slicer_bucketing;
    hr_time finish_slicer_process_buckets;
    hr_time finish_slicer_queue;
    hr_time finish_parallel_sort_cdb;

    cur_time = 0;
    double cur_time_bdgl_bucketing = 0;
    double cur_time_slicer_bucketing = 0;
    double cur_time_slicer_process_buckets = 0;
    double cur_time_slicer_queue = 0;
    double cur_time_parallel_sort_cdb = 0;
    while( it < MAX_SLICER_ITERS ) {
        start = std::chrono::high_resolution_clock::now();
        if(cdb_t[0].len<proj_error_bound){
            if(verbose) {
                std::cout << "proj_error_bound: " << proj_error_bound << std::endl;
                std::cout << it << "-th it: solution found of norm:" << cdb_t[0].len << std::endl;
            }
            finish = std::chrono::high_resolution_clock::now();
            update_curtime_slicer( cur_time, start, finish );
            ++it;
            std::cout << "slicer cur_time: " << cur_time << " it: " << it << " avg: " << cur_time/(double)(it) << std::endl;
            std::cout << "slicer cur_time_bdgl_bucketing: " << cur_time_bdgl_bucketing << " it: " << it << " avg: " << cur_time_bdgl_bucketing/(double)(it) << std::endl;
            std::cout << "slicer cur_time_slicer_bucketing: " << cur_time_slicer_bucketing << " it: " << it << " avg: " << cur_time_slicer_bucketing/(double)(it) << std::endl;
            std::cout << "slicer cur_time_slicer_process_buckets: " << cur_time_slicer_process_buckets << " it: " << it << " avg: " << cur_time_slicer_process_buckets/(double)(it) << std::endl;
            std::cout << "slicer cur_time_slicer_queue: " << cur_time_slicer_queue << " it: " << it << " avg: " << cur_time_slicer_queue/(double)(it) << std::endl;
            std::cout << "slicer cur_time_parallel_sort_cdb: " << cur_time_parallel_sort_cdb << " it: " << it << " avg: " << cur_time_parallel_sort_cdb/(double)(it) << std::endl;
            // print_siever_vect_access();
            return true;
        }

        //DO WE NEED TO DO REBUCKETING? Every X-round?
        start_bdgl_bucketing = std::chrono::high_resolution_clock::now();
        this->sieve.bdgl_bucketing(blocks, multi_hash, nr_buckets_aim, this->sieve.buckets, this->sieve.buckets_i,
                                   this->sieve.lsh_seed);
        finish_bdgl_bucketing = std::chrono::high_resolution_clock::now();
        update_curtime_slicer( cur_time_bdgl_bucketing, start_bdgl_bucketing, finish_bdgl_bucketing );

        start_slicer_bucketing = std::chrono::high_resolution_clock::now();
        slicer_bucketing(blocks, multi_hash, nr_buckets_aim, buckets, buckets_i);
        finish_slicer_bucketing = std::chrono::high_resolution_clock::now();
        update_curtime_slicer( cur_time_slicer_bucketing, start_slicer_bucketing, finish_slicer_bucketing );
        //std::cout << "slicer_bucketing finished" << std::endl;

        start_slicer_process_buckets = std::chrono::high_resolution_clock::now();
        slicer_process_buckets(buckets, buckets_i, t_queues);
        finish_slicer_process_buckets= std::chrono::high_resolution_clock::now();
        update_curtime_slicer( cur_time_slicer_process_buckets, start_slicer_process_buckets, finish_slicer_process_buckets );
        //std::cout << "slicer_process_buckets finished" << std::endl;

        start_slicer_queue = std::chrono::high_resolution_clock::now();
        slicer_queue(t_queues, transaction_db);
        finish_slicer_queue = std::chrono::high_resolution_clock::now();
        update_curtime_slicer( cur_time_slicer_queue, start_slicer_queue, finish_slicer_queue );
        //std::cout << "slicer_queue finished" << std::endl;
        
        start_parallel_sort_cdb = std::chrono::high_resolution_clock::now();
        parallel_sort_cdb();
        finish_parallel_sort_cdb = std::chrono::high_resolution_clock::now();
        update_curtime_slicer( cur_time_parallel_sort_cdb, start_parallel_sort_cdb, finish_parallel_sort_cdb );
        //std::cout << "parallel_sort_cdb finished" << std::endl;

        if(it%100==0 && verbose) {
            std::cout << "iteration " << it <<  " cdb_t[0].len " << cdb_t[0].len << " cdb_t[-1].len" << cdb_t[cdb_t.size()-1].len  << std::endl;
        }
        finish = std::chrono::high_resolution_clock::now();
        update_curtime_slicer( cur_time, start, finish );
        it++;
    }
    if(verbose) std::cerr << "Couldn't find a close vector after " << MAX_SLICER_ITERS << " iterations" << std::endl;
    std::cout << "slicer cur_time: " << cur_time << " it: " << it << " avg: " << cur_time/(double)(it) << std::endl;
    // print_siever_vect_access();
    return false;
}
