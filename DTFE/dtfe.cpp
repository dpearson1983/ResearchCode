#include <iostream>
#include <vector>
#include <iterator>
#include <cassert>
#include <algorithm>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <tpods.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned, K> Vb;
typedef CGAL::Triangulation_data_structure_3<Vb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Delaunay;
typedef Delaunay::Point Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Cell_handle Cell_handle;
typedef Delaunay::Locate_type Locate_type;
typedef Delaunay::Tetrahedron Tetrahedron;

// Function that will return an interpolated density field using the Delaunay Tessellation
void interpDTFE(std::vector<Point> &pts, vec3<double> r_min, vec3<double> L, vec3<int> N, double *nden) {
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    
    int numPts = pts.size();
    Delaunay dt(pts.begin(), pts.end());
    assert(dt.is_valid());
    
    std::vector<double> n(numPts);
    
    for (int i = 0; i < numPts; ++i) {
        Vertex_handle v = dt.nearest_vertex(pts[i]);
        std::vector<Cell_handle> c;
        dt.incident_cells(v, std::back_inserter(c));
        double Vol = 0.0;
        int numCells = c.size();
        for (int j = 0; j < numCells; ++j) {
            if (!dt.is_infinite(c[j])) {
                Tetrahedron t = dt.tetrahedron(c[j]);
                Vol += CGAL::volume(t[0], t[1], t[2], t[3]);
            }
        }
        n[i] = 4.0/Vol;
    }
    
    for (int i = 0; i < N.x; ++i) {
        double x = r_min.x + (i + 0.5)*dr.x;
        for (int j = 0; j < N.y; ++j) {
            double y = r_min.y + (j + 0.5)*dr.y;
            for (int k = 0; k < N.z; ++k) {
                double z = r_min.z + (k + 0.5)*dr.z;
                Locate_type lt;
                int li, lj;
                Point p(x, y, z);
                Cell_handle c = dt.locate(p, lt, li, lj);
                
                Tetrahedron t = dt.tetrahedron(c);
                std::vector<int> vertices(4);
                for (int i = 0; i < 4; ++i) {
                    Point pt(t[i][0], t[i][1], t[i][2]);
                    auto loc = std::find(pts.begin(), pts.end(), pt);
                    


