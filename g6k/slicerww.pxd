# -*- coding: utf-8 -*-
"""
"""

from g6k.decl cimport SlicerWW as SlicerWW_c
from g6k.siever cimport Siever

from libc.stdint cimport int16_t, int32_t, uint64_t

cdef class SlicerWW(object):
    cdef SlicerWW_c *_core
    cdef object initialized