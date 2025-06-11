//
// Created by Alexander Karenin on 12/06/2025.
//

// #ifndef G6K_WW_SLICER_H
// #define G6K_WW_SLICER_H
// #endif

#ifndef MAX_SIEVING_DIM
#define MAX_SIEVING_DIM 128
#endif

#include "compat.hpp"

class SlicerWW{

public:
    explicit SlicerWW(Siever &sieve, unsigned long int seed = 0) :
            sieve(sieve), rng(seed)
    {
        this->n = sieve.n;
        this->dbsize = sieve.db_size();
    }
    friend Siever;

    unsigned int n;
    size_t dbsize;

    Siever &sieve;
    CACHELINE_VARIABLE(rng::threadsafe_rng, rng);

    FT iterative_slice( std::array<LFT,MAX_SIEVING_DIM>& t_yr, size_t max_entries_used=0 );
    void randomize_target(std::array<LFT, MAX_SIEVING_DIM>& t_yr, size_t k );
    void randomized_iterative_slice( float* t_yr, size_t max_entries_used=0, size_t samples=1 );
    
};