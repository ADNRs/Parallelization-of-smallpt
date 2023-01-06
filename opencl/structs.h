#ifndef __SMALLPT_HEADER__
#define __SMALLPT_HEADER__

#ifdef __cplusplus
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h>
#include <cmath>
#endif

#define CAT(a, b) _CAT(a, b)
#define _CAT(a, b) a ## b

#ifdef FLOAT
#define double float
#define EPS .3
#else
#define EPS 1e-4
#endif

#ifdef __cplusplus
#define Vector CAT(cl_, CAT(double, 3))
#else
#define Vector CAT(double, 3)
#endif

#ifdef __cplusplus
inline Vector normalize(Vector v) {
    double n = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return {v.x / n, v.y / n, v.z / n};
}

inline Vector cross(Vector v1, Vector v2) {
    return {v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x};
}

inline Vector operator*(const Vector &v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}
#else
inline Vector new_vector(double x, double y, double z) {
    Vector v = {x, y, z};
    return v;
}
#endif

typedef struct Ray {
    Vector o;  // origin
    Vector d;  // direction
} Ray;

inline void rinit(Ray *r, const Vector o, const Vector d) {
    r->o = o;
    r->d = d;
}

#define DIFF 0
#define SPEC 1
#define REFR 2

typedef struct Sphere {
    double r;  // radius*radius
    Vector p;  // position
    Vector e;  // emission
    Vector c;  // color
    char t;    // reflection type
} Sphere;

#ifdef __cplusplus
inline void sinit(Sphere *s, double r, double px, double py, double pz, double ex, double ey, double ez, double cx, double cy, double cz, char t) {
    s->r = r * r;
    s->p = {px, py, pz};
    s->e = {ex, ey, ez};
    s->c = {cx, cy, cz};
    s->t = t;
}

inline double my_clamp(double x) {
    if (x < 0) return 0;
    else if (x > 1) return 1;
    else return x;
}
#endif

#endif
