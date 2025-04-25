from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix, FPLLL, config
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import ReductionError
import pickle
FPLLL.set_precision(240)

try:
  from g6k import Siever, SieverParams
  from g6k.algorithms.bkz import pump_n_jump_bkz_tour
  from g6k.utils.stats import dummy_tracer
except ImportError:
  raise ImportError("g6k not installed")

from global_consts import BKZ_SIEVING_CROSSOVER, BKZ_MAX_LOOPS, N_SIEVE_THREADS
# BKZ_SIEVING_CROSSOVER = 55

class LatticeReduction:

  def __init__(
    self,
    basis, #lattice basis to be reduced
    threads_bkz = N_SIEVE_THREADS #was 1 for primal by default
  ):

    B = IntegerMatrix.from_matrix(basis, int_type="long")

    if B.nrows <= 160:
      float_type = "long double"
    elif B.nrows <= 450:
      float_type = "dd" if config.have_qd else "mpfr"
    else:
      float_type = "mpfr"

    M = GSO.Mat(B, float_type=float_type,
      U=IntegerMatrix.identity(B.nrows, int_type=B.int_type),
      UinvT=IntegerMatrix.identity(B.nrows, int_type=B.int_type))

    M.update_gso()

    self.__bkz = BKZReduction(M)

    params_sieve = SieverParams()
    params_sieve['threads'] = threads_bkz

    self.__g6k = Siever(M, params_sieve)

  @property
  def basis(self):
    return self.__g6k.M.B
  
  @property
  def gso(self):
    return self.__g6k.M

  def BKZ(self, beta, tours=BKZ_MAX_LOOPS): #tours=8

    if beta <=  BKZ_SIEVING_CROSSOVER:
      par = BKZ_FPYLLL.Param(
        beta,
        strategies=BKZ_FPYLLL.DEFAULT_STRATEGY,
        max_loops=tours,
        flags=BKZ_FPYLLL.MAX_LOOPS
      )
      self.__bkz(par) #bkz-enum is faster this way
    else:
        for t in range(tours): #pnj-bkz is oblivious to ntours
            try:
                pump_n_jump_bkz_tour(self.__g6k, dummy_tracer, beta)
            except ReductionError as err:
                print(f"Red. err. @beta={beta} tour:{t}")
                self = LatticeReduction(self.basis)
                with open(f"badlat_{self.basis.nrows}_{beta}.pkl", "wb") as file:
                    pickle.dump( self.basis, file )
                for i in range(40,48):
                    self.BKZ(i, tours)
                pump_n_jump_bkz_tour(self.__g6k, dummy_tracer, beta)

