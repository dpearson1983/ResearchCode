#include <iostream>
#include <vector>
#include <iterator>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <galaxy.h>
#include <tpods.h>
#include "dtfe.h"

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
void interpDTFE(std::vector<galaxy<double>> &gals, vec3<double> r_min, vec3<double> L, vec3<int> N, 
                double *nden) {
    // Calculate the spacing of the grid points so that positions can be assigned.
    vec3<double> dr = {L.x/double(N.x), L.y/double(N.y), L.z/double(N.z)};
    
    // Find how many galaxies (and therefore vertices) there are. Setup storage for Point types, and
    // number density.
    int numPts = gals.size();
    std::vector<Point> pts;
    std::vector<double> n(numPts);
    
    // Populate the vector of Points, and initialize the number densities.
    std::cout << "Populating points...." << std::endl;
    for (int i = 0; i < numPts; ++i) {
        vec3<double> gal = gals[i].get_cart();
        Point p_temp(gal.x, gal.y, gal.z);
        pts.push_back(p_temp);
        n[i] = 0.0;
    }
    
    // Find the Delaunay Tessellation and make sure it's valid
    std::cout << "Creating tessellation..." << std::endl;
    Delaunay dt(pts.begin(), pts.end());
    assert(dt.is_valid());
    
    // Loop over all the galaxies, find the all cells which have a galaxy as a vertex, sum the volume
    // and then estimate the overdensity as 4/V_sum, where the factor of 4 comes from the fact that
    // each cell's volume will be counted four times (once for each vertex).
    std::cout << "Calculating number densities..." << std::endl;
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
        n[i] = (4.0*gals[i].get_weight())/Vol;
    }
    
    // Loop over all grid points, locate the tetrahedron that a grid point falls into as well as the
    // nearest vertex. The nearest vertex becomes x_0, while the remaining 3 points become x_1, x_2, and
    // x_3. Locate the associated number densities, setup matrices and vectors to solve for the components
    // of the gradient, then interpolate the field to the grid point.
//     gsl_matrix *coeff = gsl_matrix_alloc(3,3);
//     gsl_vector *gradf = gsl_vector_alloc(3);
//     gsl_vector *difff = gsl_vector_alloc(3);
//     gsl_permutation *perm = gsl_permutation_alloc(3);
//     Cell_handle c_prev; // To check for possible data reuse
    std::cout << "Interpolating to grid..." << std::endl;
// #pragma omp parallel for
    for (int i = 0; i < N.x; ++i) {
        gsl_matrix *coeff = gsl_matrix_alloc(3,3);
        gsl_vector *gradf = gsl_vector_alloc(3);
        gsl_vector *difff = gsl_vector_alloc(3);
        gsl_permutation *perm = gsl_permutation_alloc(3);
        Cell_handle c_prev; // To check for possible data reuse
        double x = r_min.x + (i + 0.5)*dr.x;
        if (omp_get_thread_num() == 0) std::cout << "Percent complete: " << (double(i)/double(N.x))*100.0 << "%" << std::endl;
        for (int j = 0; j < N.y; ++j) {
            double y = r_min.y + (j + 0.5)*dr.y;
            for (int k = 0; k < N.z; ++k) {
                double z = r_min.z + (k + 0.5)*dr.z;
                Locate_type lt;
                int li, lj;
                int index = k + N.z*(j + N.y*i);
                Point p(x, y, z);
                Cell_handle c = dt.locate(p, lt, li, lj);
                
                if (!dt.is_infinite(c)) {
                    Tetrahedron t = dt.tetrahedron(c);
                    Point x0(t[0][0], t[0][1], t[0][2]);
                    auto loc_x0 = std::find(pts.begin(), pts.end(), x0) - pts.begin();
                    if (c != c_prev) {
                        for (int l = 1; l < 4; ++l) {
                            Point xi(t[l][0], t[l][1], t[l][2]);
                            auto loc_xi = std::find(pts.begin(), pts.end(), xi) - pts.begin();
                            gsl_vector_set(difff, l - 1, n[loc_xi] - n[loc_x0]);
                            for (int m = 0; m < 3; ++m) {
                                gsl_matrix_set(coeff, l - 1, m, t[l][m] - t[0][m]);
                            }
                        }
                        int s;
                        gsl_linalg_LU_decomp(coeff, perm, &s);
                        gsl_linalg_LU_solve(coeff, perm, difff, gradf);
                    }
                    
                    nden[index] = n[loc_x0];
                    nden[index] += gsl_vector_get(gradf, 0)*(x - t[0][0]);
                    nden[index] += gsl_vector_get(gradf, 1)*(y - t[0][1]);
                    nden[index] += gsl_vector_get(gradf, 2)*(z - t[0][2]);
                    
                    c_prev = c;
                } else {
                    nden[index] = 0.0;
                }
            }
        }
        // Free the GSL structures
        gsl_matrix_free(coeff);
        gsl_vector_free(gradf);
        gsl_vector_free(difff);
        gsl_permutation_free(perm);
    }
//     // Free the GSL structures
//     gsl_matrix_free(coeff);
//     gsl_vector_free(gradf);
//     gsl_vector_free(difff);
//     gsl_permutation_free(perm);
}
