#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    double radius;
    Vector position;
    Vector emission;
    Vector color;
    ReflectType reflect_type;

    Sphere(double radius, Vector position, Vector emission, Vector color, ReflectType reflect_type) :
        radius(radius), position(position), emission(emission), color(color), reflect_type(reflect_type) {}

    double intersect(const Ray &ray) const {
        Vector dist = position - ray.origin;
        double t;
        double eps = EPS;
        double b = dist.dot(ray.direction);
        double discriminant = b*b - dist.dot(dist) + radius*radius;

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

Vector radiance(Ray ray, int depth, unsigned short *Xi) {
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
            if (erand48(Xi) < p) {
                f = f * (1/p);
            }
            else {
                return cl;
            }
        }

        cf = cf * f;

        if (obj.reflect_type == DIFF) {
            double r1 = 2 * M_PI * erand48(Xi);
            double r2 = erand48(Xi);
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

            if (erand48(Xi) < P) {
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
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps*4, 100. * y / (h - 1));

        unsigned short Xi[3] = {0, 0, static_cast<unsigned short>(y * y * y)};
        for (int x = 0; x < w; x++) {
            int i = (h - y - 1)*w + x;

            for (int sy = 0; sy < 2; sy++) {
                Vector temp;

                for (int sx = 0; sx < 2; sx++) {
                    Vector r;

                    for (int s = 0; s < samps; s++) {
                        double r1 = 2 * erand48(Xi);
                        double r2 = 2 * erand48(Xi);
                        double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        Vector d = cx*(((sx + .5 + dx)/2 + x)/w - .5)
                                 + cy*(((sy + .5 + dy)/2 + y)/h - .5)
                                 + camera.direction;

                        r = r + radiance(Ray(camera.origin + d*140, d.norm()), 0, Xi)*(1./samps);
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
