# MOAB_DIR points to top-level install dir, below which MOAB's lib/ and include/ are located
include ../makefile.config

MOAB_CXXFLAGS += -I/lus/grand/projects/gpu_hack/sigma/soft/arborx/include -I/lus/grand/projects/gpu_hack/sigma/soft/arborx/include/details
# Help Kokkos-Kernels build correctly
KOKKOS_BUILD_FLAGS= -std=c++14 -mavx -arch=sm_80  -ftree-vectorize -O3 -DNDEBUG -I/lus/grand/projects/gpu_hack/sigma/soft/kokkos-core-cuda-omp-install/include \
										-I/lus/grand/projects/gpu_hack/sigma/soft/kokkos-kernels-cuda-omp-install/include -I/grand/gpu_hack/sigma/vijaysm/install/include \
										$(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS)  --expt-extended-lambda
KOKKOS_LINK_FLAGS= -L/lus/grand/projects/gpu_hack/sigma/soft/kokkos-core-cuda-omp-install/lib  \
									 -lkokkoscontainers -lkokkoscore

MOAB_CXXFLAGS += ${KOKKOS_BUILD_FLAGS}  -I/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/include -fopenmp
MOAB_CXX = /lus/grand/projects/gpu_hack/sigma/soft/kokkos-core-cuda-omp-install/bin/nvcc_wrapper

default: all

EXAMPLES = kdtreeApp
ALLEXAMPLES = ${EXAMPLES}

all: $(ALLEXAMPLES)

kdtreeApp: kdtreeApp.o ${MOAB_LIBDIR}/libMOAB.la
	@echo "[CXXLD]  $@"
	${VERBOSE}${MOAB_CXX} -fopenmp -o $@ $< ${KOKKOS_LINK_FLAGS} ${MOAB_LIBS_LINK}  -L/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/lib -lmpi

run-kdtreeApp: kdtreeApp
	${RUNSERIAL} ./kdtreeApp

run: all $(addprefix run-,$(ALLEXAMPLES))

clean: clobber
	rm -rf ${ALLEXAMPLES}

