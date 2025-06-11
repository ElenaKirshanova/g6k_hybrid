#include <iostream>
#include <fstream>
#include <numeric>
#include <atomic>
#include <thread>
#include <mutex>

#include "siever.h"
#include "slicerww.h"
#include "fht_lsh.h"

template <typename Container, typename Container2>
void addmul_vec(Container &a, Container2 const &b, const typename Container::value_type c, int num)
{
    auto ita = a.begin();
    auto itb = b.cbegin();
    auto const ite = ita + num;

    for (; ita != ite; ++ita, ++itb)
    {
        *ita += c * (*itb);
    }
}


FT SlicerWW::iterative_slice( std::array<LFT,MAX_SIEVING_DIM>& t_yr, size_t max_entries_used ){
    if( max_entries_used == 0)
        max_entries_used = this->sieve.cdb.size();
    CompressedEntry* const fast_cdb = this->sieve.cdb.data(); // atomic load

    FT target_len = 0;
    for( size_t i = 0; i < n; i++ )
        target_len += t_yr[i] * t_yr[i];

    bool reduced = true;
    while(reduced) {
        reduced = false;

        // find best reduction
        int besti = -1;
        LFT bestk = 0;
        FT bestl = target_len;
        for (size_t j = 0; j < max_entries_used; ++j)
        {     
            // ADD POPCOUNT HERE

            int index = this->sieve.cdb[j].i;
            LFT const inner = std::inner_product(t_yr.begin(), t_yr.begin()+n, this->sieve.db[index].yr.begin(),  static_cast<LFT>(0.));

            // Test for reduction while bucketing.
            // LFT const new_l = target_len + this->sieve.fast_cdb[j].len - 2 * std::abs(inner);
            // if (UNLIKELY(new_l < bestl))
            // {   
            //     bestl = new_l;
            //     besti = index;
             LFT k = inner/fast_cdb[j].len;

            // unfortunately std::round rounds _away_ from 0
            if (k <= 0.5 && k >= -0.5) {
                k = 0;
            } else {
                k = std::round(k);
            }

            LFT new_l;
            if (UNLIKELY(k != 0.)) {
                    new_l = target_len + fast_cdb[j].len - 2 * k * inner;
                    if (UNLIKELY(new_l < bestl)) {
                        besti = index;
                        bestk = k;
                        bestl = new_l;
                    }
            }
        }

        if( besti >= 0 ) {
            reduced = true;
            // int index = besti;
            // LFT const inner = std::inner_product(t_yr.begin(), t_yr.begin()+n, this->sieve.db[index].yr.begin(),  static_cast<LFT>(0.));

            // int const sign = inner < 0 ? 1 : -1;
            // addmul_vec(t_yr,  this->sieve.db[index].yr, static_cast<LFT>(sign));
            addmul_vec(t_yr,  this->sieve.db[besti].yr, bestk, this->sieve.n);

            // recalculate length for precision
            target_len = 0;
            for( size_t i = 0; i < n; i++ ) {
                target_len += t_yr[i] * t_yr[i];
            }
        }
    }
    return target_len;
}


void SlicerWW::randomize_target(std::array<LFT, MAX_SIEVING_DIM>& t_yr, size_t k ) {
    for( size_t s = 0; s < k; s++ ) {
        // add random db element
        int index = this->sieve.cdb[rng()%this->sieve.cdb.size()].i;
        LFT* db_yr = this->sieve.db[index].yr.data();
        for( size_t i = 0; i < n; i++ )
            t_yr[i] += db_yr[i];
    }
}

void SlicerWW::randomized_iterative_slice( float* t_yr, size_t max_entries_used, size_t samples ) {
    if( max_entries_used == 0 )
        max_entries_used = this->sieve.cdb.size();
   
    // #vectors used for rerandomization
    const size_t k = 10;

    std::array<LFT, MAX_SIEVING_DIM> temp_yr;
    std::array<LFT, MAX_SIEVING_DIM> best_yr;
    FT tmp_length;
    FT best_length = 0.;
    
    for( size_t i = 0; i < n; i++ ) { 
        best_yr[i] = t_yr[i];
        best_length += best_yr[i] * best_yr[i];
    }

    for( size_t s = 0; s < samples; s++ ) {
        std::copy(best_yr.begin(), best_yr.begin()+n, temp_yr.begin());
        randomize_target( temp_yr, k );
        
        tmp_length = iterative_slice( temp_yr, max_entries_used );
        if( tmp_length < best_length ) {
            best_length = tmp_length;
            std::copy(temp_yr.begin(), temp_yr.begin()+n, best_yr.begin());
        }
    }

    for( size_t i = 0; i < n; i++ )
        t_yr[i] = best_yr[i];
}