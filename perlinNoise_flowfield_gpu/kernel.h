#pragma once
#include <vector>
#include <SFML/Graphics.hpp>

//Estructura sustitutoria para sf::Vector2f
struct float2_simple {
    float x;
    float y;
};

//Estructura para partículas del flowfield en GPU
struct ParticleGPU {
    float2_simple pos;
    float2_simple vel;
    float2_simple acc;
    float2_simple prevPos;
    float maxSpeed;
    float hue;
};

struct GpuMetrics {
    float kernelTime;
    float transferTime;
};

// Función que se llama desde main.cpp y devuelve el tiempo que tarda en generarse el flowfield en la GPU
GpuMetrics launch_cuda_flowfield(const int* h_p, const float* h_xoff, const float* h_yoff, float zoff, float2_simple* h_out, int cols, int rows);

// Función para lanzar la actualización de partículas
GpuMetrics launch_cuda_update_particles(ParticleGPU* h_particles, float2_simple* h_flowfield, int n, int cols, int rows, float scl, int width, int height);