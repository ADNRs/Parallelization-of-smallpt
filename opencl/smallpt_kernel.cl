#include "structs.h"

double sintersect(__global Sphere *s, const Ray *r) {
    Vector d = s->p - r->o;

    double t;
    double eps = EPS;
    double b = dot(d, r->d);
    double dis = b*b - dot(d, d) + s->r;

    if (dis < 0) return 0;

    dis = sqrt(dis);

    if ((t = b - dis) > eps) return t;
    else if ((t = b + dis) > eps) return t;
    else return 0;
}

int intersect(Ray *r, double *t, int *id, __global Sphere *spheres, int n) {
    double d;
    double inf = *t = 1e20;

    for (int i = 0; i < n; i++) {
        if ((d = sintersect(&spheres[i], r)) && d < *t) {
            *t = d;
            *id = i;
        }
    }

    return *t < inf;
}

double my_rand(ulong *seed) {
    *seed = (ulong)0x5deece66d**seed + 0xb & (((ulong)1 << 48) - 1);
    return (*seed >> 18) / ((double)((ulong)1 << 30));
}

Vector radiance(Ray *_ray, ulong *seed, __global Sphere *spheres, int num) {
    double t;
    int depth = 0;
    int id = 0;

    Ray ray;
    rinit(&ray, _ray->o, _ray->d);

    Vector cl = {0, 0, 0};
    Vector cf = {1, 1, 1};
    Vector res = {0, 0, 0};

    while (1) {
        if (!intersect(&ray, &t, &id, spheres, num)) {
            return res;
        }

        Sphere obj = spheres[id];

        Vector x = ray.o + ray.d*t;
        Vector n = normalize(x - obj.p);
        Vector nl = dot(n, ray.d) < 0 ? n : n * -1;
        Vector f = obj.c;

        double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;

        cl += cf * obj.e;

        if (++depth > 5) {
            if (my_rand(seed) < p) {
                f /= p;
            }
            else {
                return cl;
            }
        }

        cf *= f;

        if (obj.t == DIFF) {
            double r1 = 2 * M_PI * my_rand(seed);
            double r2 = my_rand(seed);
            double r2s = sqrt(r2);

            Vector w = nl;
            Vector u = normalize(cross(fabs(w.x) > .1 ? new_vector(0, 1, 0) : new_vector(1, 0, 0), w));
            Vector v = cross(w, u);
            Vector d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));

            rinit(&ray, x, d);
            continue;
        }
        else if (obj.t == SPEC) {
            rinit(&ray, x, ray.d - n*2*dot(n, ray.d));
            continue;
        }
        else {
            Ray reflected_ray;
            rinit(&reflected_ray, x, ray.d - n*2*dot(n, ray.d));

            char into = dot(n, nl) > 0;
            double nc = 1;
            double nt = 1.5;
            double nnt = into ? nc / nt : nt / nc;
            double ddn = dot(ray.d, nl);
            double cos2t = 1 - nnt*nnt*(1 - ddn*ddn);

            if (cos2t < 0) {
                rinit(&ray, reflected_ray.o, reflected_ray.d);
                continue;
            }

            Vector tdir = normalize(ray.d*nnt - n*((into ? 1 : -1) * (ddn*nnt + sqrt(cos2t))));

            double a = nt - nc;
            double b = nt + nc;
            double R0 = a*a / (b*b);
            double c = 1 - (into ? -ddn : dot(tdir, n));
            double Re = R0 + (1 - R0)*c*c*c*c*c;
            double Tr = 1 - Re;
            double P = .25 + .5*Re;
            double RP = Re / P;
            double TP = Tr / (1 - P);

            if (my_rand(seed) < P) {
                cf *= RP;
                rinit(&ray, reflected_ray.o, reflected_ray.d);
            }
            else {
                cf *= TP;
                rinit(&ray, x, tdir);
            }

            continue;
        }
    }
}

__kernel void render(int w,
                     int h,
                     int samps,
                     Vector cx,
                     Vector cy,
                     Ray camera,
                     __global Sphere *spheres,
                     int num,
                     __global Vector *c
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int i = (h - y - 1)*w + x;
    ulong seed = (((ulong)y << 32) | ((ulong)x << 16) | get_global_id(2));

    for (int sx = 0; sx < 2; sx++) {
        Vector temp = {0, 0, 0};

        for (int sy = 0; sy < 2; sy++) {
            Vector r = {0, 0, 0};

            for (int s = 0; s < 5; s++) {
                double r1 = 2 * my_rand(&seed);
                double r2 = 2 * my_rand(&seed);
                double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

                Vector d = cx*(((sx + .5f + dx)/2 + x)/w - .5f) + cy*(((sy + .5f + dy)/2 + y)/h - .5f) + camera.d;

                d = normalize(d);
                Vector ro = camera.o + d*140;
                Ray ray;
                rinit(&ray, ro, d);

                r += radiance(&ray, &seed, spheres, num) / samps;
            }

            temp += clamp(r, 0.f, 1.f) * .25f;
        }

        c[i] += temp;
    }

    return;
}
