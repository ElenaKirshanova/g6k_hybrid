#cython: linetrace=True
import numpy as np
from libcpp.vector cimport vector
from numpy import zeros, float32, float64, int64, matrix, array, where, matmul, identity, dot
from cysignals.signals cimport sig_on, sig_off
cimport numpy as np


from cython.operator import dereference
from decl cimport MAX_SIEVING_DIM
from decl cimport CompressedEntry, Entry_t

cdef class RandomizedSlicer(object):

    def __init__(self, Siever sieve, seed = 0):
        self._core = new RandomizedSlicer_c(dereference(sieve._core), <unsigned long>seed)

    def grow_db_with_target(self, target, size_t n_per_target):
        cdef np.ndarray target_f = zeros(MAX_SIEVING_DIM, dtype=np.float64)
        #np.ndarray[np.double_t,ndim=1] target_f

        for i in range(len(target)):
            target_f[i] = target[i]
        sig_on()
        #print("target_f:", target_f)
        self._core.grow_db_with_target(<double*> target_f.data, n_per_target)
        sig_off()

    def set_nthreads(self, size_t nt):
        self._core.set_nthreads(nt)

    def set_proj_error_bound(self, len):
        self._core.set_proj_error_bound(len)

    def set_max_slicer_interations(self, maxiter):
        self._core.set_max_slicer_interations(maxiter)

    def set_Nt(self, nt):
        self._core.set_Nt(nt)

    def set_saturation_scalar(self, sat_scalar):
        self._core.set_saturation_scalar(sat_scalar)

    def set_filename_cdbt(self, char* filename):
       self._core.set_filename_cdbt(filename)


    def bdgl_like_sieve(self, size_t nr_buckets, size_t blocks, size_t multi_hash, verbose, showstats=False):
        sig_on()
        self._core.bdgl_like_sieve(nr_buckets, blocks, multi_hash, verbose, showstats)
        sig_off()

    @property
    def _stat_get_xorpopcnt(self):
        return self._core.statistics.get_stats_xorpopcnt_total()

    @property
    def _stat_c_xorpopcnt(self):
        return self._core.statistics.collect_statistics_xorpopcnt

    @property
    def _stat_get_xorpopcnt_pass(self):
        return self._core.statistics.get_stats_xorpopcnt_pass_total()

    @property
    def _stat_c_xorpopcnt_pass(self):
        return self._core.statistics.collect_statistics_xorpopcnt_pass
    
    @property
    def _stat_get_fullscprods(self):
        return self._core.statistics.get_stats_xorpopcnt_total()

    @property
    def _stat_c_fullscprods(self):
        return self._core.statistics.collect_statistics_fullscprods
    
    @property
    def _stat_get_redsucc(self):
        return self._core.statistics.get_stats_redsucc_total()

    @property
    def _stat_c_redsucc(self):
        return self._core.statistics.collect_statistics_redsucc
    
    @property
    def _stat_get_replacements(self):
        return self._core.statistics.get_stats_replacements()

    @property
    def _stat_c_replacements(self):
        return self._core.statistics.collect_statistics_replacements
    
    @property
    def _stat_get_collisions(self):
        return self._core.statistics.get_stats_collisions()

    @property
    def _stat_c_collisions(self):
        return self._core.statistics.collect_statistics_collisions
    
    @property
    def _stat_get_reds_during_randomization(self):
        return self._core.statistics.get_stats_reds_during_randomization()

    @property
    def _stat_c_reds_during_randomization(self):
        return self._core.statistics.collect_statistics_reds_during_randomization
    
    @property
    def _stat_get_buck_over_max(self):
        return self._core.statistics.get_stats_buck_over_max()

    @property
    def _stat_c_buck_over_max(self):
        return self._core.statistics.collect_statistics_buck_over_max
    
    @property
    def _stat_get_buck_over_num(self):
        return self._core.statistics.get_stats_buck_over_num()

    @property
    def _stat_c_buck_over_num(self):
        return self._core.statistics.collect_statistics_buck_over_num
    
    @property
    def _stat_get_itercount_slicer(self):
        return self._core.statistics.get_stats_itercount_slicer()

    @property
    def _stat_c_itercount_slicer(self):
        return self._core.statistics.collect_statistics_itercount_slicer
    
        # This dictionary controls how statistics are exported / displayed.
    #
    # Format is as follows: key equals the C++ = decl.pxd = _stat_get_ name
    # Value is [SequenceID, short description, long description, algs, OPTIONAL: repr]
    # where  SequenceID is a number used to determine in which order we write output
    #        short description is the prefix used in (short) humand-readable output
    #        long description is a meaningful "docstring"
    #        algs is a set of algorithms where this statistic is meaningful
    #        repr is optional and is passed to the Accumulator inside the TreeTracer as its
    #           repr argument. Set to "max" to output the max value instead of the sum.

    all_statistics = {
        "xorpopcnt"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}],
        "xorpopcnt_pass"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}],
        "fullscprods"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}],
        "redsucc"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}],
        "replacements"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}],
        "collisions"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}],
        "reds_during_randomization"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}],
        "buck_over_max"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}],  
        "buck_over_num"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                                 {"bdgl2"}], 
        "itercount_slicer"            : [10,  "XPC   :",  "total number of xorpopcnt calculations",                            {"bdgl2"}], 
     }

    @property
    def stats(self):
        "Returns all collected statistics of the current sieve as a dictionary"
        ret = {"cdb_t-size:" : self._core.cdb_t.size()}
        for key in RandomizedSlicer.all_statistics:
            if(getattr(self,"_stat_c_" + key) == True):
                ret[key] = getattr(self, "_stat_get_" + key)
        return ret

    def itervalues_cdb_t(self,return_with_index=True):
        """
        Iterate over all entries in the target database (in the order determined by the compressed database cdb_t)

        """
        cdef Entry_t *e;

        for i in range(self._core.cdb_t.size()):
            e = &self._core.db_t[self._core.cdb_t[i].i]
            r = [e.yr[j] for j in range(self._core.n)]
            index = e.i
            if return_with_index:
                yield ( tuple(r), index )
            else:
                raise NotImplementedError

