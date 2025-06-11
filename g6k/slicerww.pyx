#cython: linetrace=True
import numpy as np
from libcpp.vector cimport vector
from numpy import zeros, float32, float64, int64, matrix, array, where, matmul, identity, dot
from cysignals.signals cimport sig_on, sig_off
cimport numpy as np


from cython.operator import dereference
from decl cimport MAX_SIEVING_DIM
from decl cimport CompressedEntry, Entry_t

cdef class SlicerWW(object):

    def __init__(self, Siever sieve, seed = 0):
        self._core = new SlicerWW_c(dereference(sieve._core), <unsigned long>seed)

    # CVPP
    # Target is expected in normalized gram schmidth coordinates, like the internal yr of the db
    def randomized_iterative_slice(self, target_yr, size_t samples, size_t max_entries_used=0):
        # assert(self.initialized)
        if max_entries_used == 0:
            max_entries_used = self._core.dbsize
        assert(max_entries_used <= self._core.dbsize)

        print(f"n: {self._core.n}")
        print(f"dbsize: {self._core.dbsize}")

        cdef np.ndarray t_yr = zeros( (self._core.n,), dtype=float32)

        for i in xrange(self._core.n):
            t_yr[i] = target_yr[i]

        sig_on()
        self._core.randomized_iterative_slice( <float*> t_yr.data, max_entries_used, samples )
        sig_off()

        return_yr = zeros( (self._core.n,), dtype=float32)

        for i in xrange(self._core.n):
            return_yr[i] = t_yr[i]
        return return_yr