/*
    The implementation of `double my_rand(unsigned long long)` and
    `double my_rand(unsigned long long, double [])` referred to glibc
    sources `stdlib/drand48-iter.c` and `stdlib/erand48_r.c`, and thus
    this source file is licensed under LGPL 3.0.

    The original program is written by Kevin Beason, and the original
    license is provided in the root directory as required.
*/

#include <cassert>
#include <cmath>
#include <cstdio>

#ifdef FLOAT
#define double float
#define EPS .3
#else
#define EPS 1e-4
#endif

struct Vector {
    double x;
    double y;
    double z;

    Vector(double x=0, double y=0, double z=0) : x(x), y(y), z(z) {}

    friend Vector operator+(const Vector &a, const Vector &b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    friend Vector operator-(const Vector &a, const Vector &b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    friend Vector operator*(const Vector &a, const Vector &b) {
        return {a.x * b.x, a.y * b.y, a.z * b.z};
    }

    friend Vector operator*(const Vector &a, const double mul) {
        return {a.x * mul, a.y * mul, a.z * mul};
    }

    Vector & norm() {
        return *this = *this * (1 / sqrt(x*x + y*y + z*z));
    }

    double dot(const Vector &other) const {
        return x*other.x + y*other.y + z*other.z;
    }

    Vector cross(const Vector &other) const {
        return {y*other.z-z*other.y, z*other.x - x*other.z, x*other.y - y*other.x};
    }
};

struct Ray {
    Vector origin;
    Vector direction;

    Ray(Vector origin, Vector direction) : origin(origin), direction(direction) {}
};

enum ReflectType {DIFF, SPEC, REFR};

struct Sphere {
    double radius2;
    Vector position;
    Vector emission;
    Vector color;
    ReflectType reflect_type;

    Sphere(double radius, Vector position, Vector emission, Vector color, ReflectType reflect_type) :
        radius2(radius*radius), position(position), emission(emission), color(color), reflect_type(reflect_type) {}

    double intersect(const Ray &ray) const {
        Vector dist = position - ray.origin;
        double t;
        double eps = EPS;
        double b = dist.dot(ray.direction);
        double discriminant = b*b - dist.dot(dist) + radius2;

        if (discriminant < 0) return 0;

        discriminant = sqrt(discriminant);

        if ((t = b - discriminant) > eps) return t;
        else if ((t = b + discriminant) > eps) return t;
        else return 0;
    }
};

Sphere spheres[] = {
    Sphere(1e5,  {1e5+1, 40.8, 81.6},   {},           {.75, .25, .25},    DIFF),  // left
    Sphere(1e5,  {-1e5+99, 40.8, 81.6}, {},           {.25, .25, .75},    DIFF),  // right
    Sphere(1e5,  {50, 40.8, 1e5},       {},           {.75, .75, .75},    SPEC),  // back
    Sphere(1e5,  {50, 40.8, -1e5+170},  {},           {.75, .75, .75},    SPEC),  // front
    Sphere(1e5,  {50, 1e5, 81.6},       {},           {.75, .75, .75},    DIFF),  // bottom
    Sphere(1e5,  {50, -1e5+81.6, 81.6}, {},           {.75, .75, .75},    DIFF),  // top
    Sphere(16.5, {27, 16.5, 47},        {},           {.999, .999, .999}, REFR),
    Sphere(13.7, {73, 13.7, 78},        {},           {.999, .999, .999}, REFR),
    Sphere(7.3,  {50, 7.3, 103},        {},           {.999, .999, .999}, REFR),
    Sphere(600,  {50, 681.6-.27, 81.6}, {12, 12, 12}, {},                 DIFF),
};

double clamp(double x) {
    if (x < 0) return 0;
    else if (x > 1) return 1;
    else return x;
}

int to_int(double x) {
    return static_cast<int>(pow(clamp(x), 1/2.2)*255 + 0.5);
}

bool intersect(const Ray &ray, double &t, int &id) {
    int n = sizeof spheres / sizeof (Sphere);
    double d;
    double inf = t = 1e20;

    for (int i = 0; i < n; i++) {
        if ((d = spheres[i].intersect(ray)) && d < t) {
            t = d;
            id = i;
        }
    }

    return t < inf;
}

#ifdef FLOAT
double my_rand(unsigned long long &seed) {
    seed = 0x5deece66dUL*seed + 0xb & ((1L << 48) - 1);
    return (seed >> 18) / ((double)(1L << 30));
}
#else
#include <immintrin.h>
#include <ieee754.h>

union packed_int {
    unsigned long long ull;
    struct {
        unsigned int ui1;
        unsigned int ui0;
    } ui;
    struct {
        unsigned short us3;
        unsigned short us2;
        unsigned short us1;
        unsigned short us0;
    } us;
};

double my_rand(unsigned long long &seed) {
    packed_int seeds;
    unsigned short a;
    unsigned short b;
    unsigned short c;
    ieee754_double temp;

    seeds.ull = seed = 0x5deece66dULL*seed + 0xb;

    a = seeds.us.us3;
    b = seeds.us.us2;
    c = seeds.us.us1;

    temp.ieee.negative = 0;
    temp.ieee.exponent = IEEE754_DOUBLE_BIAS;
    temp.ieee.mantissa0 = (c << 4) | (b >> 12);
    temp.ieee.mantissa1 = ((b & 0xfff) << 20) | (a << 4);

    return temp.d - 1.0;
}

void my_rand2(unsigned long long &seed, double res[]) {
    packed_int seeds[2];
    packed_int maskfff;
    packed_int mantissa0;
    packed_int mantissa1;

    __m64 vseeds;
    __m64 va;
    __m64 vb;
    __m64 vc;
    __m64 vmantissa0;
    __m64 vmantissa1;
    __m64 vmaskfff;
    __m64 clsft4;
    __m64 brsft12;
    __m64 blsft20;
    __m64 alsft4;

    ieee754_double temp[2];

    seeds[0].ull = seed = 0x5deece66dULL*seed + 0xb;
    seeds[1].ull = seed = 0x5deece66dULL*seed + 0xb;
    maskfff.ui.ui0 = maskfff.ui.ui1 = 0xfff;

    vmaskfff = _mm_cvtsi64_m64(maskfff.ull);
    va = _mm_cvtsi64_m64(((unsigned long long)seeds[0].us.us3 << 32) | seeds[1].us.us3);
    vb = _mm_cvtsi64_m64(((unsigned long long)seeds[0].us.us2 << 32) | seeds[1].us.us2);
    vc = _mm_cvtsi64_m64(((unsigned long long)seeds[0].us.us1 << 32) | seeds[1].us.us1);

    clsft4 = _m_pslldi(vc, 4);
    brsft12 = _m_psrldi(vb, 12);
    vmantissa0 = _m_por(clsft4, brsft12);

    blsft20 = _m_pand(vb, vmaskfff);
    blsft20 = _m_pslldi(blsft20, 20);
    alsft4 = _m_pslldi(va, 4);
    vmantissa1 = _m_por(blsft20, alsft4);

    mantissa0.ull = _mm_cvtm64_si64(vmantissa0);
    mantissa1.ull = _mm_cvtm64_si64(vmantissa1);

    temp[0].ieee.negative = 0;
    temp[1].ieee.negative = 0;
    temp[0].ieee.exponent = IEEE754_DOUBLE_BIAS;
    temp[1].ieee.exponent = IEEE754_DOUBLE_BIAS;
    temp[0].ieee.mantissa0 = mantissa0.ui.ui0;
    temp[1].ieee.mantissa0 = mantissa0.ui.ui1;
    temp[0].ieee.mantissa1 = mantissa1.ui.ui0;
    temp[1].ieee.mantissa1 = mantissa1.ui.ui1;

    res[0] = temp[0].d - 1.0;
    res[1] = temp[1].d - 1.0;
}
#endif

Vector radiance(Ray ray, int depth, unsigned long long &seed) {
    double t;
    int id = 0;
    Vector cl(0, 0, 0);
    Vector cf(1, 1, 1);

    while (1) {
        if (!intersect(ray, t, id)) return Vector();

        const Sphere &obj = spheres[id];

        Vector x = ray.origin + ray.direction*t;
        Vector n = (x - obj.position).norm();
        Vector nl = n.dot(ray.direction) < 0 ? n : n * -1;
        Vector f = obj.color;
        double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;

        cl = cl + cf*obj.emission;

        if (++depth > 5) {
            if (my_rand(seed) < p) {
                f = f * (1/p);
            }
            else {
                return cl;
            }
        }

        cf = cf * f;

        if (obj.reflect_type == DIFF) {
            double rands[2];

            #ifdef FLOAT
            rands[0] = my_rand(seed);
            rands[1] = my_rand(seed);
            #else
            my_rand2(seed, rands);
            #endif

            double r1 = 2 * M_PI * rands[0];
            double r2 = rands[1];
            double r2s = sqrt(r2);

            Vector w = nl;
            Vector u = (fabs(w.x) > .1 ? Vector(0, 1) : Vector(1)).cross(w).norm();
            Vector v = w.cross(u);
            Vector d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).norm();

            ray = Ray(x, d);
            continue;
        }
        else if (obj.reflect_type == SPEC) {
            ray = Ray(x, ray.direction - n*2*n.dot(ray.direction));
            continue;
        }
        else {
            Ray reflacted_ray(x, ray.direction - n*2*n.dot(ray.direction));
            bool into = n.dot(nl) > 0;
            double nc = 1;
            double nt = 1.5;
            double nnt = into ? nc / nt : nt / nc;
            double ddn = ray.direction.dot(nl);
            double cos2t = 1 - nnt*nnt*(1 - ddn*ddn);

            if (cos2t < 0) {
                ray = reflacted_ray;
                continue;
            }

            Vector tdir = (ray.direction*nnt - n*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
            double a = nt - nc;
            double b = nt + nc;
            double R0 = a*a / (b*b);
            double c = 1 - (into ? -ddn : tdir.dot(n));
            double Re = R0 + (1 - R0)*c*c*c*c*c;
            double Tr = 1 - Re;
            double P = .25 + .5*Re;
            double RP = Re / P;
            double TP = Tr / (1 - P);

            if (my_rand(seed) < P) {
                cf = cf * RP;
                ray = reflacted_ray;
            }
            else {
                cf = cf * TP;
                ray = Ray(x, tdir);
            }

            continue;
        }
    }
}

int main(int argc, char *argv[]) {
    assert(argc == 2);

    int w = 1024;
    int h = 768;
    int samps = atoi(argv[1]) / 4;
    Ray camera(Vector(50, 52, 295.6), Vector(0, -0.042612, -1).norm());
    Vector cx(w*.5135 / h);
    Vector cy = cx.cross(camera.direction).norm() * .5135;
    Vector *c = new Vector[w*h]();

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; y++) {
        unsigned long long seed = ((unsigned long long)y * y * y);

        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps*4, 100. * y / (h - 1));

        for (int x = 0; x < w; x++) {
            int i = (h - y - 1)*w + x;

            for (int sy = 0; sy < 2; sy++) {
                Vector temp;

                for (int sx = 0; sx < 2; sx++) {
                    Vector r;

                    for (int s = 0; s < samps; s++) {
                        double rands[2];

                        #ifdef FLOAT
                        rands[0] = my_rand(seed);
                        rands[1] = my_rand(seed);
                        #else
                        my_rand2(seed, rands);
                        #endif

                        double r1 = 2 * rands[0];
                        double r2 = 2 * rands[1];
                        double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        Vector d = cx*(((sx + .5 + dx)/2 + x)/w - .5)
                                 + cy*(((sy + .5 + dy)/2 + y)/h - .5)
                                 + camera.direction;

                        r = r + radiance(Ray(camera.origin + d*140, d.norm()), 0, seed)*(1./samps);
                    }

                    temp = temp + Vector(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
                }

                c[i] = c[i] + temp;
            }
        }
    }

    FILE *f = fopen("image.ppm", "w");

    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0 ; i < w * h; i++) {
        fprintf(f, "%d %d %d ", to_int(c[i].x), to_int(c[i].y), to_int(c[i].z));
    }
}
