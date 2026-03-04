#pragma once
//#include <SFML/System/Vector2.hpp> // Para que reconozca sf::Vector2f

struct float2_simple {
    float x;
    float y;
};

// Función que se llama desde main.cpp
void launch_cuda_flowfield(const int* h_p, const float* h_xoff, const float* h_yoff, float zoff, float2_simple* h_out, int cols, int rows);