# MOAB_DIR points to top-level install dir, below which MOAB's lib/ and include/ are located
include ../makefile.config

default: all

EXAMPLES = spmvApp
ALLEXAMPLES = ${EXAMPLES}

# If using Kokkos-Kernels or Ginkgo, need C++14 standard; Also disable some verbose warnings.
MOAB_CXXFLAGS += -std=c++14 -Wno-delete-non-abstract-non-virtual-dtor -Wno-sign-compare -Wno-unused-parameter

# Help Eigen3 perform better on host by linking against OpenMP runtime
MOAB_CXXFLAGS += -Xclang -fopenmp
MOAB_LIBS_LINK += -lomp

all: $(ALLEXAMPLES)

spmvApp: spmvApp.o netcdf.o ncvalues.o spmvApp-eigen3.h  spmvApp-ginkgo.h spmvAppUtils.h ${MOAB_LIBDIR}/libMOAB.la
	@echo "[CXXLD]  $@"
	${VERBOSE}${MOAB_CXX} -o $@ $< netcdf.o ncvalues.o ${MOAB_LIBS_LINK}

run-spmvApp: spmvApp
	${RUNSERIAL} ./spmvApp

run: all $(addprefix run-,$(ALLEXAMPLES))

clean: clobber
	rm -rf ${ALLEXAMPLES}

