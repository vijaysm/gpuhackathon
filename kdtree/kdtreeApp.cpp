/** @example kdtree_arborx_test.cpp
 * \brief This example shows how to perform local point-in-element searches with MOAB's
 * kdtree searching functionality, and with newer GPU capable ArborX  searching functionality.
 *
 * MOAB's SpatialLocator functionality performs point-in-element searches over a local or parallel
 * mesh. SpatialLocator is flexible as to what kind of tree is used and what kind of element basis
 * functions are used to localize elements and interpolate local fields.
 *
 * Usage:
 *  Default mesh: ./kdtreeApp -d 3 -r 2
 *  Custom mesh: ./kdtreeApp -d 3 -i mesh_file.h5m
 */

// C++ includes
#include <iostream>
#include <chrono>

#include "moab/Core.hpp"
#include "moab/Interface.hpp"
#include "moab/Range.hpp"
#include "moab/CN.hpp"
#include "moab/ProgOptions.hpp"
#include "moab/NestedRefine.hpp"

#include "moab/AdaptiveKDTree.hpp"
//#include "TestUtil.hpp"

#include <ArborX.hpp>
#include <ArborX_Version.hpp>
#include <Kokkos_Core.hpp>
#include <random>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

using namespace moab;
using namespace std;

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::high_resolution_clock::time_point Timer;
using std::chrono::duration_cast;

Timer start;
std::map< std::string, std::chrono::nanoseconds > timeLog;

#define PUSH_TIMER()          \
    {                         \
        start = Clock::now(); \
    }

#define POP_TIMER( EventName )                                                                                \
    {                                                                                                         \
        std::chrono::nanoseconds elapsed = duration_cast< std::chrono::nanoseconds >( Clock::now() - start ); \
        timeLog[EventName]               = elapsed;                                                           \
    }

#define PRINT_TIMER( EventName )                                                                                       \
    {                                                                                                                  \
        std::cout << "[ " << EventName                                                                                 \
                  << " ]: elapsed = " << static_cast< double >( timeLog[EventName].count() / 1e6 ) << " milli-seconds" \
                  << std::endl;                                                                                        \
    }

int main( int argc, char** argv )
{
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = ExecutionSpace::memory_space;

    std::string test_file_name  = string( "meshfiles/3k-tri-sphere.vtk" );
    int num_queries             = 1000000;
    int dimension               = 2;
    int uniformRefinementLevels = 0;
    int max_depth = 30;
    int max_per_leaf = 6;
    std::ostringstream options;


    ProgOptions opts;

    // Need option handling here for input filename
    opts.addOpt< int >( "dim,d", "Dimension of the problem and mesh (default=2)", &dimension );
    opts.addOpt< int >( "queries,n", "Number of queries to perform on the mesh (default=1E4)", &num_queries );
    opts.addOpt< int >( "refine,r", "Number of levels of uniform refinements to perform on the mesh (default=0)",
                        &uniformRefinementLevels );
    opts.addOpt< int > ("max_depth,m", "Maximum depth in kdtree (default 30)", &max_depth);
    opts.addOpt <int >("max_per_leaf,l", "Maximum per leaf (default 6) ", &max_per_leaf);
    opts.addOpt< std::string >( "file,i", "File name to load the mesh)", &test_file_name );

    opts.parseCommandLine( argc, argv );

    // Instantiate
    ErrorCode rval;
    Core mb;
    EntityHandle baseFileset, fileset;

    rval = mb.create_meshset( MESHSET_SET, baseFileset );MB_CHK_ERR( rval );

    PUSH_TIMER()
    // Load the file
    rval = mb.load_file( test_file_name.c_str(), &baseFileset );MB_CHK_SET_ERR( rval, "Error loading file" );

    cout << "load file:  " << test_file_name << "\n";
    if( uniformRefinementLevels )
    {
        moab::NestedRefine uref( &mb, nullptr, baseFileset );
        std::vector< int > uniformRefinementDegree( uniformRefinementLevels, 2 );
        std::vector< EntityHandle > level_sets;
        rval = uref.generate_mesh_hierarchy( uniformRefinementLevels,
                                                uniformRefinementDegree.data(),
                                                level_sets, true );MB_CHK_ERR( rval );
        assert( (int)level_sets.size() == uniformRefinementLevels + 1 );
        fileset = level_sets[uniformRefinementLevels];
    }
    else
        fileset = baseFileset;
    POP_TIMER( "MeshIO-Refine" )
    PRINT_TIMER( "MeshIO-Refine" )

    // Get all elements in the file
    Range elems;
    rval = mb.get_entities_by_dimension( fileset, dimension, elems );MB_CHK_SET_ERR( rval, "Error getting 3d elements" );

    cout << "mesh has " << elems.size()  << " cells \n";
    PUSH_TIMER()
    // Create a tree to use for the location service
    // Can we accelerate this setup phase on a GPU as well ??
    // Or may be use Kokkos/OpenMP for CPU executor ?
    AdaptiveKDTree kd( &mb );
    EntityHandle tree_root;
    rval = mb.create_meshset(MESHSET_SET, tree_root); MB_CHK_ERR( rval );
    options << "MAX_DEPTH=" << max_depth << ";";
    options << "MAX_PER_LEAF=" << max_per_leaf << ";";

    FileOptions file_opts( options.str().c_str() );
    rval                   = kd.build_tree( elems, &tree_root, &file_opts );MB_CHK_ERR( rval );
    POP_TIMER( "MOAB KdTree-Setup" )
    PRINT_TIMER( "MOAB KdTree-Setup" )

    // Get the box extents
    CartVect min, max, box_extents, pos;
    unsigned  depth;
    kd.get_info( tree_root, &min[0], &max[0], depth );

    cout << "box: " << min << " " <<  max << "depth: " << depth << "\n";
    // print various info about the tree
    kd.print();
    BoundBox box (min,max);
    box_extents  = 1.1 * (max - min);

    std::vector<CartVect> queries;
    queries.reserve(num_queries);
    for( int i = 0; i < num_queries; i++ )
    {
        pos  = box.bMin + CartVect( box_extents[0] * .01 * ( rand() % 100 ), box_extents[1] * .01 * ( rand() % 100 ),
                                   box_extents[2] * .01 * ( rand() % 100 ) );
        // project on sphere of radius 100
        pos = pos/pos.length()*100;
        queries.push_back(pos);
    }
    PUSH_TIMER()

    EntityHandle leaf_out;
    for( int i = 0; i < num_queries; i++ )
    {
        pos  = queries[i];

        // project on a sphere if dim 2?
        rval = kd.point_search( &pos[0], leaf_out);  MB_CHK_ERR( rval );

    }
    POP_TIMER( "KdTree-Query" )
    PRINT_TIMER( "KdTree-Query" )


    PUSH_TIMER()
    Kokkos::ScopeGuard guard(argc, argv);

    std::cout << "ArborX version: " << ArborX::version() << std::endl;
    std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;


    // Create the View for the bounding boxes, on device
    Kokkos::View<ArborX::Box*, ExecutionSpace::memory_space> bounding_boxes("bounding_boxes", elems.size());
    // with MemorySpace=Kokkos::CudaSpace, BoundingVolume=ArborX::Box, Enable=void
    //Kokkos::View<ArborX::Box*> bounding_boxes("bounding_boxes", elems.size());
    // mirror view on host, will be populated
    auto h_bounding_boxes = Kokkos::create_mirror_view(bounding_boxes);
    //points.reserve(elems.size());

    std::vector<CartVect>  coords;
    coords.resize(27);// max possible
    for (size_t i=0; i<elems.size(); i++)
    {
        EntityHandle cell=elems[i];
        const EntityHandle *conn ;
        int nnodes;
        rval = mb.get_connectivity(cell, conn, nnodes); MB_CHK_ERR( rval );
        rval = mb.get_coords(conn, nnodes, &coords[0][0] );MB_CHK_SET_ERR( rval, "can't get coordinates" );
        for (int j=0; j<nnodes; j++)
        {
            ArborX::Details::expand(h_bounding_boxes(i),
                    ArborX::Point{ (float)coords[j][0], (float)coords[j][1], (float)coords[j][2]}  );
        }
    }
    Kokkos::deep_copy(bounding_boxes, h_bounding_boxes);

    // Create the bounding volume hierarchy
    ArborX::BVH<Kokkos::CudaSpace> bvh(ExecutionSpace{}, bounding_boxes);


    POP_TIMER( "ArborX-Build" )
    PRINT_TIMER( "ArborX-Build" )

    PUSH_TIMER()
    // queries
    // copy from paper

    // Create the View for the spatial-based queries
    // Kokkos::View<decltype(ArborX::intersects(ArborX::Box{})) *, DeviceType>
    Kokkos::View< decltype(ArborX::intersects(ArborX::Box{})) * , ExecutionSpace::memory_space> queries_ar("queries", num_queries);
    // Fill in the queries on host mirror, then copy to device ?

    auto h_queries_ar = create_mirror_view(queries_ar);

    /*
    Kokkos::parallel_for("fill_queries",
                           Kokkos::RangePolicy<ExecutionSpace>(0, num_queries ),
                           KOKKOS_LAMBDA(int i) {
                                 h_queries_ar(i) = ArborX::intersects(
                                         ArborX::Box (
                                         ArborX::Point{ (float)queries[i][0], (float)queries[i][1], (float)queries[i][2]} ,
                                         ArborX::Point{ (float)queries[i][0], (float)queries[i][1], (float)queries[i][2]}
                                         )

                                     );
                           });
                           */

    for( int i = 0; i < num_queries; i++ )
    {
        pos  = queries[i];
        h_queries_ar(i) = ArborX::intersects(
                                                 ArborX::Box (
                                                 { (float)pos[0], (float)pos[1], (float)pos[2] } ,
                                                 { (float)pos[0], (float)pos[1], (float)pos[2] }
                                                 )
                                              );
    }
    // copy from host to device
    Kokkos::deep_copy(queries_ar, h_queries_ar);

    // Perform the search
    Kokkos::View<int*, ExecutionSpace::memory_space> offsets("offset", 0);
    Kokkos::View<int*, ExecutionSpace::memory_space> indices("indices", 0);
    ArborX::query(bvh, ExecutionSpace{}, queries_ar, indices, offsets);

    POP_TIMER( "ArborX-Query" )
    PRINT_TIMER( "ArborX-Query" )
    // end copy
    //
    /*for (Range::iterator it=elems.begin(); it!= elems.end(); it++)
    {
        EntityHandle cell=*it; // should we get the box

        points.push_back(;

    }*/


    cout << bvh.size() << "\n";

    ArborX::Box bb = bvh.bounds();
    ArborX::Point maxcorn = bb.maxCorner();
    cout << "max_corner: " << maxcorn[0] <<" " <<  maxcorn[1] << " "  << maxcorn[2] << "\n";

 /*   // query at the same locations
    PUSH_TIMER()

    POP_TIMER( "ArborX-Query" )
    PRINT_TIMER( "ArborX-Query" )
*/


    return 0;
}

