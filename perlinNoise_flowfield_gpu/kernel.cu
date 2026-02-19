#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

// --- FUNCIONES AUXILIARES (__device__) ---
//Función que hace que la mezcla ocurra suave al principio y al final
__device__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); // => 6t^5 - 15t^4 = 10t^3
}

//Función para interpolación lineal -> Mezcla entra a y b usando un parámetro t entre 0 y 1 
__device__ float lerp(float t, float a, float b) {
    return a + t * (b - a); //Es una recta; no tiene suavizado en los extremos. Si t crece linealmente, el resultado cambia linealmente
}

//Función que genera el valor de gradiente en una esquina del cubo
__device__ float grad(int hash, float x, float y, float z) {
    int h = hash & 15; //convierte un número grande entre 0 y 15 (16 posibles direcciones; se pueden repetir pero con distinto signo)
    float u = h < 8 ? x : y; //se elige una dirección pseudoaleataoria
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z); //se calcula el producto escalar
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v); //se genera y devuelve el valor de gradiente en una esquina del cubo
}

//Función que calcula el ruido en un punto 3D
__device__ float baseNoise(const int* p, float x, float y, float z) {
    //Se calcula en qué celda o cubo caen las coordenas x, y, z. & 255 hace que si nos salimos de 255, volvamos a empezar desde 0 (hace que el ruido sea infinito y repetitivo)
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;
    int Z = (int)floorf(z) & 255;

    //Nos quedamos solo con la parte decimal. En el anterior cálculo nos quedamos con el cubo en el que estamos (parte entera), y ahora con la parte decimal:
    // La dirección (X, Y o Z) te dice hacia qué pared te estás moviendo y la parte decimal te dice cuánto te has alejado de la pared de origen.
    // por ej. en la posición 5.3, estamos en el cubo 5 y a 0.3 de distancia del borde "izquierdo" (o el que sea) del cubo
    x -= floorf(x);
    y -= floorf(y);
    z -= floorf(z);

    //Llamamos a una curva de suavizado para no movernos de forma lineal (brusca) y hacer que los cambios de dirección sean suaves y orgánicos
    float u = fade(x);
    float v = fade(y);
    float w = fade(z);

    //Búsqueda en cadena en la tabla de permutación para localizar las esquinas del cubo.
    // Usamos X para elegir cara (A/B), Y para elegir altura (AA, AB, BA, BB).
    int A = p[X] + Y; //izquierda
    int AA = p[A] + Z; //esquina inferior, izquierda, fondo
    int AB = p[A + 1] + Z; //esquina superior, izquierda, fondo

    int B = p[X + 1] + Y; //derecha
    int BA = p[B] + Z; //esquina inferior, derecha, fondo
    int BB = p[B + 1] + Z; //esquina superior, derecha, fondo

    //Se calcula la influencia de las 8 esquinas del cubo sobre nuestro punto y las mezcla todas (interpola) para darnos un valor final único.
    // (se obtiene el ID con la tabla p y en base a él, se obtiene la dirección con grad, y se hace que se muevan con los decimales que calculamos antes y la función lerp).
    float res = lerp(w,
        // Mezcla de las 4 esquinas de la cara trasera (Z)
        lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1.0f, y, z)),
            lerp(u, grad(p[AB], x, y - 1.0f, z), grad(p[BB], x - 1.0f, y - 1.0f, z))),
        // Mezcla de las 4 esquinas de la cara delantera (Z + 1)
        lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1.0f), grad(p[BA + 1], x - 1.0f, y, z - 1.0f)),
            lerp(u, grad(p[AB + 1], x, y - 1.0f, z - 1.0f), grad(p[BB + 1], x - 1.0f, y - 1.0f, z - 1.0f)))
    );

    return (res + 1.0f) * 0.5f; //El cáculo matemático suele dar valores entre -1 y 1. Como nosotros queremos algo más fácil de manejar (como un color), lo convertimos a un rango de 0 a 1
}

// --- KERNEL ---
__global__ void flowfield_kernel(const int* d_p, const float* d_xoff, const float* d_yoff, const float zoff, float2* d_out, int cols, int rows) {
    // Calculamos el ID del hilo global en 2D
    size_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Comprobamos que no está fuera del límite
    if (x_idx < cols && y_idx < rows) {

        float x = d_xoff[x_idx]; //Accedemos y guardamos el valor del ruido x en la posición x_idx 
        float y = d_yoff[y_idx]; //Accedemos y guardamos el valor del ruido y en la posición y_idx
        // !! d_xoff y d_yoff son números decimales (0.01, 0.02...) sirven para consultar el mapa de ruido de forma suave y obtener un diseño orgánico.

        // Lógica Fractal (4 octavas)
        float total = 0; //acumulador donde se van sumando todas las capas del ruido
        float freq = 1.0f; //Variable que controla el detalle -> Si es baja (1.0), el ruido es grande y suave (como colinas); si
        //es alta (sube en cada vuelta), el ruido se vuelve pequeño y nervioso (como rocas o arena)
        float amp = 1.0f; //fuerza de la capa
        float maxV = 0; //máximo valor posible -> guarda cuánto suman las capas si todas dieran su valor máximo

        for (int i = 0; i < 4; i++) { //se suman las capas de detalle (4 octavas -> 4 veces)
            //La primera capa hace las formas grandes, la segunda los detalles medianos, la tercera los pequeños...
            total += baseNoise(d_p, x * freq, y * freq, zoff * freq) * amp; //sumamos el ruido actual multiplicado por su fuerza
            maxV += amp; //Vamos sumando el máximo valor posible para luego poder normalizar.
            amp *= 0.5f; //en cada vuelta bajamos a la mitad la fuerza de la capa para que los detalles pequeños no tapen a las formas grandes
            freq *= 2.0f; //Se sube el nivel de detalle para la siguiente capa. Al aumentar la frecuencia, el ruido se repite más rápido creando detalles más finos
        }

        float n = total / maxV; //Como hemos sumado varias capas, el resultado almacenado en total puede ser muy grande. Dividiéndolo por maxV nos aseguramos de que
        // el resultado final esté entre 0 y 1
        float angle = n * 6.283185f * 4.0f; //Convertimos ese valor en una dirección. 6.283185f es 2pi, y al multiplicar por 4 permitimos que el ruido tenga suficiente rango para dar giros complejos

        //Calculamos el índice único para la coordenada actual (x,y) para guardar el resultado en el flowfield de la GPU (para luego pasarlo a la CPU)
        size_t idx = x_idx + y_idx * cols; //esta fórmula convierte la coordenada (columna, fila) en una posición lineal única en el array.
        
        //Convertimos el ángulo en un vector unitario
        d_out[idx].x = cosf(angle); //guardamos cuánto empuja a la derecha/izquierda
        d_out[idx].y = sinf(angle); //guardamos cúanto empuja hacia arriba/abajo
    }
}

// --- FUNCIÓN DE LANZAMIENTO ---
void launch_cuda_flowfield(const int* h_p, const float* h_xoff, const float* h_yoff, float zoff, sf::Vector2f* h_out, int cols, int rows) {

    // 1. Tamaños y Bytes
    const size_t TOTAL_CELLS = (size_t)cols * rows; //búmero total de celdas
    const size_t FLOW_BYTES = TOTAL_CELLS * sizeof(float2); //cada celda guarda un vector de 2 floats, por lo que el total de bytes guardados será de 2 * sizeof(float) * núm. de celdas total
    const size_t PERM_BYTES = 512 * sizeof(int); //la tabla de permutación de Perlin tiene 512 elementos de tipo int, por lo que ocupará 512 * sozepf(int)
    const size_t XOFF_BYTES = cols * sizeof(float); //cada columna tienen un offset X de tipo float que ocupará el número de columnas * sizeof(float)
    const size_t YOFF_BYTES = rows * sizeof(float); //cada fila tiene un offset Y de tipo float que ocupará el número de filas * sizeof(float)

    // 2. Declaramos los punteros de memoria en la GPU (Device)
    float2* d_out = NULL; //Declaración de puntero para array flowfield de salida
    int* d_p = NULL; //Declaración de puntero para array de tabla de permutación de Perlin
    float* d_xoff = NULL; //Declaración de puntero para array de offset en x (columnas)
    float* d_yoff = NULL; //Declaración de puntero para array de offset en y (filas)

    // 3. Reservamos memoria en la GPU
    cudaMalloc(&d_out, FLOW_BYTES); //se reserva lo que ocupa en total toda la rejilla del flowfield (cada celda con un vector de dos valores)
    cudaMalloc(&d_p, PERM_BYTES); //reserva del espacio necesario para la tabla de permutación de Perlin
    cudaMalloc(&d_xoff, XOFF_BYTES); //reserva del espacio necesario para los offset en x (columnas)
    cudaMalloc(&d_yoff, YOFF_BYTES); //reserva del espacio necesario para los offset en y (filas)

    // 4. Transferimos los datos desde el host (CPU) a la GPU (Device)
    cudaMemcpy(d_p, h_p, PERM_BYTES, cudaMemcpyHostToDevice); //se copian los datos de h_p a d_p que ocupan PER_BYTES
    cudaMemcpy(d_xoff, h_xoff, XOFF_BYTES, cudaMemcpyHostToDevice); //se copian los datos de los offsets de h_xoff a d_xoff que ocupan XOFF_BYTES
    cudaMemcpy(d_yoff, h_yoff, YOFF_BYTES, cudaMemcpyHostToDevice); //se copian los datos de los offsets de h_yoff a d_yoff que ocupan YOFF_BYTES

    // 5. Lanzamos el kernel (Configuración de rejilla 2D)
    dim3 THREADS_PER_BLOCK(16, 16, 1); //número de hilos en cada bloque -> 16 en x, 16 en y, ninguno en z -> 16*16 = 256
    dim3 BLOCKS_PER_GRID( //el número de bloques por grid será:
        (size_t)ceil((float)cols / THREADS_PER_BLOCK.x),  //ceil(número total de columnas / número de hilos en x)
        (size_t)ceil((float)rows / THREADS_PER_BLOCK.y) //ceil(número total de filas / números de hilos en y)
    );

    flowfield_kernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (d_p, d_xoff, d_yoff, zoff, d_out, cols, rows); //lanzamiento del kernel

    // 6. Copiamos el array de resultados desde la GPU al host
    cudaMemcpy(h_out, d_out, FLOW_BYTES, cudaMemcpyDeviceToHost);

    // 7. Se libera la memoria reservada en la GPU
    cudaFree(d_out);
    cudaFree(d_p);
    cudaFree(d_xoff);
    cudaFree(d_yoff);

    d_out = NULL;
    d_p = NULL;
    d_xoff = d_yoff = NULL; //se pueden igualar ambos a NULL a la vez porque son del mismo tipo
}