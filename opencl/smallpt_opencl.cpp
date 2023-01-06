#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "structs.h"

int to_int(double x) {
    return static_cast<int>(pow(my_clamp(x), 1/2.2)*255 + 0.5);
};

cl_kernel kernel = nullptr;
cl_program program = nullptr;
cl_mem s_buf = nullptr;
cl_mem c_buf = nullptr;
cl_command_queue queue = nullptr;
cl_context context = nullptr;

int _main(int argc, char *argv[]) {
    assert(argc == 2);

    Sphere spheres[10];

    sinit(&spheres[0],    1e5,     1e5+1, 40.8, 81.6,      0, 0, 0,       .75, .25, .25,       DIFF);  // left
    sinit(&spheres[1],    1e5,     -1e5+99, 40.8, 81.6,    0, 0, 0,       .25, .25, .75,       DIFF);  // right
    sinit(&spheres[2],    1e5,     50, 40.8, 1e5,          0, 0, 0,       .75, .75, .75,       SPEC);  // back
    sinit(&spheres[3],    1e5,     50, 40.8, -1e5+170,     0, 0, 0,       .75, .75, .75,       SPEC);  // front
    sinit(&spheres[4],    1e5,     50, 1e5, 81.6,          0, 0, 0,       .75, .75, .75,       DIFF);  // bottom
    sinit(&spheres[5],    1e5,     50, -1e5+81.6, 81.6,    0, 0, 0,       .75, .75, .75,       DIFF);  // top
    sinit(&spheres[6],    16.5,    27, 16.5, 47,           0, 0, 0,       .999, .999, .999,    REFR);
    sinit(&spheres[7],    13.7,    73, 13.7, 78,           0, 0, 0,       .999, .999, .999,    REFR);
    sinit(&spheres[8],    7.3,     50, 7.3, 103,           0, 0, 0,       .999, .999, .999,    REFR);
    sinit(&spheres[9],    600,     50, 681.6-.27, 81.6,    12, 12, 12,    0, 0, 0,             DIFF);

    int w = 1024;
    int h = 768;
    int samps = atoi(argv[1]) / 4;
    int n = sizeof spheres / sizeof (Sphere);

    assert(samps % 5 == 0);

    Ray camera;
    rinit(&camera, {50.f, 52.f, 295.6f}, normalize({0.f, -0.042612f, -1.f}));

    Vector cx = {0.f, 0.f, 0.f};
    cx.x = w * .5135f / h;

    Vector cy = normalize(cross(cx, camera.d));
    cy.x *= .5135f; cy.y *= .5135f; cy.z *= .5135f;

    Vector *c = (Vector *)malloc(sizeof (Vector) * w * h);

    cl_int err;

    // initialize device
    cl_uint platforms_num;
    err = clGetPlatformIDs(0, nullptr, &platforms_num);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs: %d\n", err);
        return 1;
    }

    assert(platforms_num > 0);

    cl_platform_id platforms[platforms_num];
    err = clGetPlatformIDs(platforms_num, platforms, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs: %d\n", err);
        return 1;
    }

    // get the first devices of the first platform
    cl_uint devices_num;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &devices_num);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs: %d\n", err);
        return 1;
    }

    assert(devices_num > 0);

    cl_device_id devices[devices_num];
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, devices_num, devices, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs: %d\n", err);
        return 1;
    }

    // create context
    context = clCreateContext(nullptr, devices_num, devices, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext: %d\n", err);
        return 1;
    }

    // read the kernel file
    std::ifstream kernel_file("smallpt_kernel.cl");
    std::string code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
    char *source = new char[code.size() + 1]();
    source[code.size()] = '\n';
    strcpy(source, code.c_str());

    program = clCreateProgramWithSource(context, 1, (const char **)&source, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource: %d\n", err);
        return 1;
    }

    // build kerenl
    err = clBuildProgram(program, 1, devices,
    #ifdef FLOAT
        "-DFLOAT"
    #else
        nullptr
    #endif
        , nullptr, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram: %d\n", err);

        size_t log_size;
        err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetProgramBuildInfo: %d\n", err);
            return 1;
        }

        char *log = new char[log_size];
        err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetProgramBuildInfo: %d\n", err);
            return 1;
        }

        fprintf(stderr, "%s\n", log);
    }

    // create buffers
    s_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_HOST_NO_ACCESS|CL_MEM_COPY_HOST_PTR, sizeof spheres, spheres, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer: %d\n", err);
        return 1;
    }

    c_buf = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_HOST_READ_ONLY, sizeof (Vector) * w * h, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer: %d\n", err);
        return 1;
    }

    // initialize the kernel function
    kernel = clCreateKernel(program, "render", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 0, sizeof w, &w);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 1, sizeof h, &h);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 2, sizeof samps, &samps);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 3, sizeof cx, &cx);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 4, sizeof cy, &cy);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 5, sizeof camera, &camera);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 6, sizeof (cl_mem), &s_buf);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 7, sizeof n, &n);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 8, sizeof (cl_mem), &c_buf);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", err);
        return 1;
    }

    // create queue
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE|CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueueWithProperites: %d\n", err);
        return 1;
    }

    // set work size
    size_t global_size[] = {(size_t)w, (size_t)h, (size_t)samps / 5};
    size_t local_size[] = {16, 12, 1};

    // start calculation
    fprintf(stderr, "Host: Started");
    err = clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global_size, local_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel: %d\n", err);
        return 1;
    }

    err = clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, sizeof (Vector) * w * h, c, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer: %d\n", err);
        return 1;
    }

    err = clFlush(queue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clFlush: %d\n", err);
        return 1;
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clFinish: %d\n", err);
        return 1;
    }

    // write image
    FILE *f = fopen("image.ppm", "w");

    fprintf(stderr, "\rHost: Finished (%d spp)", (samps / 5 * 5 * 4));

    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0 ; i < w * h; i++) {
        fprintf(f, "%d %d %d ", to_int(c[i].x), to_int(c[i].y), to_int(c[i].z));
    }

    return 0;
}

int main(int argc, char *argv[]) {
    int ret_val = _main(argc, argv);

    if (kernel) {
        clReleaseKernel(kernel);
    }
    if (program) {
        clReleaseProgram(program);
    }
    if (s_buf) {
        clReleaseMemObject(s_buf);
    }
    if (c_buf) {
        clReleaseMemObject(c_buf);
    }
    if (queue) {
        clReleaseCommandQueue(queue);
    }
    if (context) {
        clReleaseContext(context);
    }

    return ret_val;
}
