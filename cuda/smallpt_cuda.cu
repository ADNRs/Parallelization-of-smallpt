#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

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

    __host__ __device__ Vector(double x=0, double y=0, double z=0) : x(x), y(y), z(z) {}

    __host__ __device__ friend Vector operator+(const Vector &a, const Vector &b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }

    __host__ __device__ friend Vector operator-(const Vector &a, const Vector &b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }

    __host__ __device__ friend Vector operator*(const Vector &a, const Vector &b) {
        return {a.x * b.x, a.y * b.y, a.z * b.z};
    }

    __host__ __device__ friend Vector operator*(const Vector &a, const double mul) {
        return {a.x * mul, a.y * mul, a.z * mul};
    }

    __host__ __device__ Vector & norm() {
        return *this = *this * (1 / sqrt(x*x + y*y + z*z));
    }

    __host__ __device__ double dot(const Vector &other) const {
        return x*other.x + y*other.y + z*other.z;
    }

    __host__ __device__ Vector cross(const Vector &other) const {
        return {y*other.z-z*other.y, z*other.x - x*other.z, x*other.y - y*other.x};
    }
};

struct Ray {
    Vector origin;
    Vector direction;

    __host__ __device__ Ray(Vector origin, Vector direction) : origin(origin), direction(direction) {}
};

enum ReflectType {DIFF, SPEC, REFR};

struct Sphere {
    double radius;
    Vector position;
    Vector emission;
    Vector color;
    ReflectType reflect_type;

    __host__ __device__ Sphere(double radius, Vector position, Vector emission, Vector color, ReflectType reflect_type) :
        radius(radius), position(position), emission(emission), color(color), reflect_type(reflect_type) {}

    __device__ double intersect(const Ray &ray) const {
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

__host__ __device__ double clamp(double x) {
    if (x < 0) return 0;
    else if (x > 1) return 1;
    else return x;
}

int to_int(double x) {
    return static_cast<int>(pow(clamp(x), 1/2.2)*255 + 0.5);
}

__device__ bool intersect(const Ray &ray, double &t, int &id, Sphere *spheres, int n) {
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

__device__ double my_rand(unsigned long long &seed) {
    seed = (unsigned long long)0x5deece66d*seed + 0xb & (((unsigned long long)1 << 48) - 1);
    return (seed >> 18) / ((double)((unsigned long long)1 << 30));
}

__device__ Vector radiance(Ray ray, unsigned long long &seed, Sphere *spheres, int num) {
    double t;
    int depth = 0;
    int id = 0;
    Vector cl(0, 0, 0);
    Vector cf(1, 1, 1);

    while (1) {
        if (!intersect(ray, t, id, spheres, num)) return Vector();

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
            double r1 = 2 * M_PI * my_rand(seed);
            double r2 = my_rand(seed);
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

__global__ void render(int w,
                       int h,
                       int samps,
                       Vector cx,
                       Vector cy,
                       Ray camera,
                       Sphere *spheres,
                       int num,
                       Vector *c
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int i = (h - y - 1)*w + x;
    unsigned long long seed = (((unsigned long long)y << 32) | ((unsigned long long)x << 16) | blockIdx.z*blockDim.z + threadIdx.z);

    for (int sx = 0; sx < 2; sx++) {
        Vector temp{0, 0, 0};

        for (int sy = 0; sy < 2; sy++) {
            Vector r{0, 0, 0};

            for (int s = 0; s < 5; s++) {
                double r1 = 2 * my_rand(seed);
                double r2 = 2 * my_rand(seed);
                double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

                Vector d = cx*(((sx + .5 + dx)/2 + x)/w - .5) + cy*(((sy + .5 + dy)/2 + y)/h - .5) + camera.direction;

                d.norm();
                Ray ray(camera.origin + d*140, d);

                r = r + radiance(ray, seed, spheres, num)*(1. / samps);
            }

            temp = temp + Vector(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
        }

        c[i] = c[i] + temp;
    }
}

int main(int argc, char *argv[]) {
    assert(argc == 2);

    int w = 1024;
    int h = 768;
    int samps = atoi(argv[1]) / 4;
    int n = sizeof spheres / sizeof (Sphere);

    assert(samps % 5 == 0);

    Ray camera(Vector(50, 52, 295.6), Vector(0, -0.042612, -1).norm());
    Vector cx(w*.5135 / h);
    Vector cy = cx.cross(camera.direction).norm() * .5135;
    Vector *c_host = new Vector[w*h]();
    Vector *c_device;
    Sphere *s_device;

    dim3 grid_size(w / 16, h / 12, samps / 5);
    dim3 block_size(16, 12, 1);

    cudaError_t err;

    err = cudaMalloc(&c_device, sizeof (Vector) * w * h);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc(&s_device, sizeof spheres);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(s_device, spheres, sizeof spheres, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(err));
        return 1;
    }

    render<<<grid_size, block_size>>>(w, h, samps, cx, cy, camera, s_device, n, c_device);
    cudaDeviceSynchronize();

    err = cudaMemcpy(c_host, c_device, sizeof (Vector) * w * h, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(err));
        return 1;
    }

    FILE *f = fopen("image.ppm", "w");

    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0 ; i < w * h; i++) {
        fprintf(f, "%d %d %d ", to_int(c_host[i].x), to_int(c_host[i].y), to_int(c_host[i].z));
    }

    cudaFree(c_device);
    cudaFree(s_device);
}
