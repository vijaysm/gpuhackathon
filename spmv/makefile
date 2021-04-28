# MOAB_DIR points to top-level install dir, below which MOAB's lib/ and include/ are located
include ../makefile.config

default: all

EXAMPLES = spmvApp
ALLEXAMPLES = ${EXAMPLES}

# If using Kokkos-Kernels or Ginkgo, need C++14 standard; Also disable some verbose warnings.
MOAB_CXXFLAGS += -g3 -mavx  -DUSE_EIGEN3 -DUSE_GINKGO -std=c++14 -Wno-delete-non-abstract-non-virtual-dtor -Wno-sign-compare -Wno-unused-parameter -Wno-narrowing

all: $(ALLEXAMPLES)

spmvApp.o: spmvApp.cpp spmvApp-eigen3.h spmvApp-ginkgo.h spmvAppUtils.h ${MOAB_LIBDIR}/libMOAB.la

spmvApp: spmvApp.o netcdf.o ncvalues.o ${MOAB_LIBDIR}/libMOAB.la
	@echo "[CXXLD]  $@"
	${VERBOSE}${MOAB_CXX} -o $@ $< netcdf.o ncvalues.o ${MOAB_LIBS_LINK} \
		-L/lus/grand/projects/gpu_hack/sigma/vijaysm/install/lib \
		-lginkgo -lginkgo_device -lginkgo_omp -lginkgo_cuda -lginkgo_reference -lginkgo_hip -lginkgo_dpcpp # Ginkgo LIBS

# Help Kokkos-Kernels build correctly
KOKKOS_BUILD_FLAGS= -DUSE_KOKKOS -mavx -arch=sm_80  -ftree-vectorize -O3 -DNDEBUG -I/lus/grand/projects/gpu_hack/sigma/soft/kokkos-core-cuda-omp-install/include \
										-I/lus/grand/projects/gpu_hack/sigma/soft/kokkos-kernels-cuda-omp-install/include -I/grand/gpu_hack/sigma/vijaysm/install/include \
										$(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS)  --expt-extended-lambda
KOKKOS_LINK_FLAGS= -L/lus/grand/projects/gpu_hack/sigma/soft/kokkos-core-cuda-omp-install/lib -L/lus/grand/projects/gpu_hack/sigma/soft/kokkos-kernels-cuda-omp-install/ \
									 -lkokkoskernels -lkokkoscontainers -lkokkoscore

spmvApp-Kokkos: spmvApp-Kokkos.o netcdf.o ncvalues.o spmvApp-kokkos.h spmvAppUtils.h ${MOAB_LIBDIR}/libMOAB.la
	@echo "[CXXLD]  $@"
	/lus/grand/projects/gpu_hack/sigma/soft/kokkos-core-cuda-omp-install/bin/nvcc_wrapper $(KOKKOS_LDFLAGS) ${KOKKOS_BUILD_FLAGS} -isystem /grand/gpu_hack/sigma/vijaysm/install/include \
		-I/grand/gpu_hack/sigma/vijaysm/install     -I/grand/gpu_hack/sigma/vijaysm/install/include -isystem /grand/gpu_hack/sigma/vijaysm/install/include -I/grand/gpu_hack/sigma/vijaysm/install \
		-DMESH_DIR=\""/grand/gpu_hack/sigma/vijaysm/moab/build/gpuhack/../../MeshFiles/unittest"\" -fopenmp spmvApp-Kokkos.o netcdf.o ncvalues.o -o spmvApp-Kokkos \
		-L/lus/grand/projects/gpu_hack/sigma/soft/kokkos-core-cuda-omp-install/lib -L/lus/grand/projects/gpu_hack/sigma/soft/kokkos-kernels-cuda-omp-install/lib \
		-lkokkoskernels -lkokkoscontainers -lkokkoscore ${MOAB_LIBS_LINK} -L/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/lib -lmpi -lcusparse -lcublas

spmvApp-Kokkos.o: spmvApp.cpp spmvApp-kokkos.h 
	@echo "[CXX] $@"
	${VERBOSE}/lus/grand/projects/gpu_hack/sigma/soft/kokkos-core-cuda-omp-install/bin/nvcc_wrapper ${KOKKOS_BUILD_FLAGS}  -isystem /grand/gpu_hack/sigma/vijaysm/install/include -I/grand/gpu_hack/sigma/vijaysm/install     -I/grand/gpu_hack/sigma/vijaysm/install/include -isystem /grand/gpu_hack/sigma/vijaysm/install/include -I/grand/gpu_hack/sigma/vijaysm/install  -I/grand/gpu_hack/sigma/vijaysm/install/include   -I/grand/gpu_hack/sigma/vijaysm/install/include -DMESH_DIR=\""/grand/gpu_hack/sigma/vijaysm/moab/build/gpuhack/../../MeshFiles/unittest"\" -std=c++17 -fopenmp -c spmvApp.cpp  -o spmvApp-Kokkos.o

run-spmvApp: spmvApp
	${RUNSERIAL} ./spmvApp

run: all $(addprefix run-,$(ALLEXAMPLES))

clean: clobber
	rm -rf ${ALLEXAMPLES}

