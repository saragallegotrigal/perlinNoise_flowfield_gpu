#include <SFML/Graphics.hpp> //dibujar ventanas, líneas, colores, maejar eventos, etc.
#include <cstdint>
#include <vector> //guardar listas dinámicas
#include <random> //para posiciones iniciales aleatorias
#include <cmath> //sin, cos, sqrt, floor
#include <algorithm> //shiffle, iota
#include <numeric>
#include <cuda_runtime.h>

//medir tiempo:
#include <chrono>
#include <iostream>

//kernel.h para kernel
#include "kernel.h"

// ------------------------ Perlin Noise (Improved + Fractal) ------------------------
struct Perlin3D {
    std::vector<int> p;

    Perlin3D(unsigned seed) { 
        p.resize(256); //tamaño del vector de puntos p
        std::iota(p.begin(), p.end(), 0); //se llena el vector (0, 1, 2... 255)
        std::mt19937 rng(seed); //se CREA el generador de números aleatorios y lo inicializa con la seed 1337
        std::shuffle(p.begin(), p.end(), rng); //se mezclan los valores en cuanto al generador iniciado en rng
        //shuffle convierte los números aleatorios de rng para las posiciones en valores válidos dentro del rango del vector p
        p.insert(p.end(), p.begin(), p.end()); //se añaden los elementos ya mezclados al final aumentando el tamaño total a 512
    }

    //función que suaviza las transiciones y hace que no haya "cortes". Sin ella, el ruido tendría esquinas duras
    static float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }

    //función para interpolación lineal: entre a y b, dame el punto t
    static float lerp(float t, float a, float b) { return a + t * (b - a); }

    //Función que convierte un número (hash) en una dirección y calcula su influencia
    static float grad(int hash, float x, float y, float z) {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
    }

    // Ruido Perlin base (1 octava)
    float baseNoise(float x, float y, float z) const {
        int X = (int)std::floor(x) & 255;
        int Y = (int)std::floor(y) & 255;
        int Z = (int)std::floor(z) & 255;

        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);
        float u = fade(x), v = fade(y), w = fade(z);

        int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
        int B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;

        float res = lerp(w,
            lerp(v,
                lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))
            ),
            lerp(v,
                lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))
            )
        );
        return (res + 1.0f) * 0.5f; // mapea de [-1,1] a [0,1]
    }

    // *** NUEVA FUNCIÓN ***: Ruido Fractal (simula p5.js noise)
    // p5.js usa por defecto 4 octavas y un falloff de 0.5
    // Aquí ya no es Perlin “puro”, es Perlin acumulado.
    float noise(float x, float y, float z, int octaves = 4, float persistence = 0.5f) const {
        float total = 0;
        float frequency = 1;
        float amplitude = 1;
        float maxValue = 0;  // Usado para normalizar el resultado a 0.0 - 1.0

        for (int i = 0; i < octaves; i++) {
            total += baseNoise(x * frequency, y * frequency, z * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= 2; // La frecuencia se duplica en cada octava
        }

        return total / maxValue;
    }
};

// ------------------------ Utilities ------------------------

//Función que convierte color: hue (ángulo), saturación, brillo, alpha... para que las partículas tengan color suave
static sf::Color HSVtoRGBA(float h, float s, float v, float a) {
    float H = std::fmod(h, 360.f) / 60.f;
    float S = s / 255.f;
    float V = v / 255.f;
    int i = (int)std::floor(H);
    float f = H - i;
    float p = V * (1.f - S);
    float q = V * (1.f - S * f);
    float t = V * (1.f - S * (1.f - f));
    float r = 0, g = 0, b = 0;
    switch (i) {
    case 0: r = V; g = t; b = p; break;
    case 1: r = q; g = V; b = p; break;
    case 2: r = p; g = V; b = t; break;
    case 3: r = p; g = q; b = V; break;
    case 4: r = t; g = p; b = V; break;
    default: r = V; g = p; b = q; break;
    }
    return sf::Color((uint8_t)(r * 255), (uint8_t)(g * 255), (uint8_t)(b * 255), (uint8_t)a);
}

//Función que evita que una velocidad crezca indefinidamente.
static void limit(sf::Vector2f& v, float maxMag) {
    float m2 = v.x * v.x + v.y * v.y;
    if (m2 > maxMag * maxMag) {
        float m = std::sqrt(m2);
        float k = maxMag / (m + 1e-9f);
        v.x *= k; v.y *= k;
    }
}

// ------------------------ Particle ------------------------
struct Particle {

    //Cada partícula tiene: una posición, velocidad, aceleración y posición anterior (previa) EN CADA FRAME
    sf::Vector2f pos;
    sf::Vector2f vel{ 0.f,0.f };
    sf::Vector2f acc{ 0.f,0.f };
    sf::Vector2f prevPos;
    float maxSpeed = 2.0f;
    float hue;

    //Constructor de una partícula
    Particle(float x, float y, float hueDeg) : pos(x, y), prevPos(x, y), hue(hueDeg) {}

    //Función para aplicarle una aceleración (fuerza): Permite acumular todas las fuerzas que actúan sobre la partícula en este frame
    void applyForce(const sf::Vector2f& f) { acc += f; }

    //Función actualizar
    void update() {
        vel += acc; //se suma la aceleración
        limit(vel, maxSpeed); //se limita su velocidad para que no sobrepase la velocidad máxima definida
        pos += vel; //se actualiza la posición añadiendo la velocidad
        acc *= 0.f; //se resetea la aceleración para poder usarla luego en el SIGUIENTE FRAME
    }

    //Función que actualiza la posición previa de la partícula a la actual para usarla en el SIGUIENTE FRAME
    void updatePrev() { prevPos = pos; } //prevPos se usa para dibujar líneas desde la posición anterior hasta la actual
    // Sin prevPos no habría estela, solo puntos individuales

    /*
     *
     Función que detecta si la partícula sale de la ventana (WIDTH x HEIGHT), y la “envuelve” al lado contrario,
     actualizando prevPos si se ha envuelto para evitar líneas muy largas. Es necesaria porque mantiene a las partículas
     siempre dentro de la ventana, evitando líneas raras que atraviesan toda la pantalla cuando se salen. Si no existiese,
     las partículas desaparecerían fuera de la ventana o dibujarían líneas gigantes de un extremo al otro
     *
     */
    void edges(int W, int H) {
        bool wrapped = false;
        if (pos.x > W) { pos.x = 0; wrapped = true; }
        if (pos.x < 0) { pos.x = (float)W; wrapped = true; }
        if (pos.y > H) { pos.y = 0; wrapped = true; }
        if (pos.y < 0) { pos.y = (float)H; wrapped = true; }
        if (wrapped) updatePrev();
    }


    //Convierte la posición (pos.x, pos.y) en un índice del flowfield, calculando en qué celda de la rejilla está la partícula.
    int index(int cols, int rows, float scl) const { //scl es los píxeles que tiene cada celda (10x10 por ej.)
        // 1. Divide por scl para convertir la posición en “coordenadas de celda”
        // 2. Aplica floor para obtener la celda como número entero
        int x = (int)std::floor(pos.x / scl); //fila
        int y = (int)std::floor(pos.y / scl); //columna

        // 3. Clamp para evitar seg faults si la partícula toca el borde exacto
        // Se comprueba que las coordenadas estén dentro del número de filas y columnas del flowfield
        if (x >= cols) x = cols - 1;
        if (y >= rows) y = rows - 1;
        // Si son coordenadas negativas, se establecen como cero
        if (x < 0) x = 0;
        if (y < 0) y = 0;

        // 4. Conversión de coordenadas 2D (x, y) a índice lineal
        return x + y * cols;
    }
};

// ------------------------ Función flowfield en CPU------------------------
void flowfield_cpu(Perlin3D& perlin, const float inc, const int cols, const int rows, std::vector<sf::Vector2f>& flowfield, float& zoff) {
    // --- Generación del Flow Field ---
    //std::cout << "EJECUCION EN CPU" << std::endl;
    float yoff = 0.f; //coordenada y en el ruido perlin

    //Recorremos la rejilla FILA A FILA
    for (int y = 0; y < rows; ++y) {
        float xoff = 0.f; //coordenada x en el ruido perlin
        //dentro de cada FILA, recorremos cada COLUMNA
        for (int x = 0; x < cols; ++x) {
            // n es un valor entre 0 y 1 (lo devuelve la función perlin.noise(), con el punto actual y con un número
            // de octavas = 4 (capas que van añadiendo detalle), con 0,5 de persistencia (las capas van teniendo la
            // mitad de influencia que la anterior)
            float n = perlin.noise(xoff, yoff, zoff, 4, 0.5f);

            float angle = n * 6.28318530718f * 4.f; //se convierte el ruido en un ángulo (2pi = 6.28318; *4 para más
            // giros y curvas). Así, angle es un ángulo en radianes: 0 pi derecha, pi/2 pi abajo, pi pi izquierda y 3pi/2 pi arriba
            sf::Vector2f v(std::cos(angle), std::sin(angle)); //vector con la dirección del viento en
            // esa celda del flow field (cos es cuánto apunta en X, sin es cuánto apunta en Y)
            // NOTA: setMag(1) está implícito porque cos/sin crean vector unitario (que siempre tienen longitud 1)

            //Guardamos el vector en el array 1D
            flowfield[x + y * cols] = v;

            //Avanzamos en el ruido en x
            xoff += inc;
        }
        //Avanzamos en el ruido en y
        yoff += inc;
    }
    //Avanzamos el tiempo para animar lentamente el flowfield
    //zoff += 0.0003f;
}

// ============================================================================
// APARTADO 2.2: VALIDACIÓN TEMPORAL (Propagación del Error T=1 hasta T=10000)
// ============================================================================

void realizar_test_error(int frames_objetivo, int N_test, int cols, int rows, float inc, unsigned seed, Perlin3D& perlin, std::vector<float>& xoff_matrix, std::vector<float>& yoff_matrix) {

    // 1. Inicialización idéntica para ambos
    std::vector<Particle> particles_cpu;
    std::vector<Particle> particles_gpu;
    std::mt19937 rng_test(seed);
    std::uniform_real_distribution<float> rx(0.f, 960.f);
    std::uniform_real_distribution<float> ry(0.f, 540.f);

    for (int i = 0; i < N_test; ++i) {
        float x = rx(rng_test), y = ry(rng_test);
        particles_cpu.emplace_back(x, y, 0);
        particles_gpu.emplace_back(x, y, 0);
    }

    std::vector<sf::Vector2f> ff_cpu(cols * rows);
    std::vector<sf::Vector2f> ff_gpu(cols * rows);
    float zoff_test = 0.0f;

    // 2. Bucle de simulación hasta el frame objetivo
    for (int t = 1; t <= frames_objetivo; t++) {
        // Generar Flowfields
        flowfield_cpu(perlin, inc, cols, rows, ff_cpu, zoff_test);
        launch_cuda_flowfield(perlin.p.data(), xoff_matrix.data(), yoff_matrix.data(), zoff_test, reinterpret_cast<float2_simple*>(ff_gpu.data()), cols, rows);

        // Actualizar CPU
        for (auto& p : particles_cpu) {
            int idx = p.index(cols, rows, 10.f);
            p.applyForce(ff_cpu[idx]);
            p.update();
            p.edges(960, 540);
        }
        // Actualizar GPU
        for (auto& p : particles_gpu) {
            int idx = p.index(cols, rows, 10.f);
            p.applyForce(ff_gpu[idx]);
            p.update();
            p.edges(960, 540);
        }
        zoff_test += 0.001f;
    }

    // 3. Cálculo de métricas al final del periodo T
    double sumErrorSq = 0;
    double sumDiff = 0;
    float maxDiff = 0;


    for (int i = 0; i < N_test; i++) {
        float dx = particles_cpu[i].pos.x - particles_gpu[i].pos.x;
        float dy = particles_cpu[i].pos.y - particles_gpu[i].pos.y;
        float dist = std::sqrt(dx * dx + dy * dy);

        sumDiff += dist;
        sumErrorSq += (dist * dist);
        if (dist > maxDiff) maxDiff = dist;
    }


    std::cout << "RESULTADOS PARA T = " << frames_objetivo << ":" << std::endl;
    std::cout << "Seed = " << seed << std::endl;
    std::cout << "- Dif. Media: " << sumDiff / N_test << " px" << std::endl;
    std::cout << "- Dif. Maxima: " << maxDiff << " px" << std::endl;
    std::cout << "- RMSE: " << std::sqrt(sumErrorSq / N_test) << std::endl;
    std::cout << "--------------------------------------" << std::endl;
}

void update_particles_cpu(std::vector<Particle>& particles, const std::vector<sf::Vector2f>& flowfield, sf::VertexArray& lines, int cols, int rows, float scl, int WIDTH, int HEIGHT)
{
    for (std::size_t i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];

        // 1. Obtener fuerza del flowfield y aplicar
        int idx = p.index(cols, rows, scl);
        p.applyForce(flowfield[idx]);

        // 2. Actualizar física y bordes
        p.update();
        p.edges(WIDTH, HEIGHT);

        // 3. Actualizar el VertexArray para SFML
        std::size_t vIndex = i * 2;

        // Evitar líneas que cruzan la pantalla (Wrap-around check)
        float dx = std::abs(p.pos.x - p.prevPos.x);
        float dy = std::abs(p.pos.y - p.prevPos.y);

        sf::Color c = sf::Color(234, 137, 154, 50); // Color rosita suave

        if (dx > 50.f || dy > 50.f) {
            lines[vIndex].position = p.pos;
            lines[vIndex + 1].position = p.pos;
        }
        else {
            lines[vIndex].position = p.prevPos;
            lines[vIndex + 1].position = p.pos;
        }

        lines[vIndex].color = c;
        lines[vIndex + 1].color = c;

        // 4. Preparar para el siguiente frame
        p.updatePrev();
    }
}

// ------------------------ Función principal ------------------------
int main() {
    const int WIDTH = 960; //ancho de la ventana
    const int HEIGHT = 540; //alto de la ventana
    const float inc = 0.1f; //cuánto se avanza en el espacio del ruido
    // (más pequeño -> ruido más suave y curvas más fluidas; más grande -> ruido más brusco con cambios más fuertes)
    const float scl = 10.f; //tamaño de cada celda del flowfield en píxeles -> cada vector del flowfield controla un cuadrado de nxn píxeles: 5, 10, 15, 20, 30
    const int cols = (int)std::floor(WIDTH / scl); //número de columnas redondeado hacia abajo: floor(960/10)
    const int rows = (int)std::floor(HEIGHT / scl); //número de filas redondeado hacia abajo: floor(540/10)
    const size_t flowCount = (size_t)cols * (size_t)rows; //número total de vectores del flowfield (filas*columnas)

    //Semilla
    unsigned seed = 1337; //1337, 2026, 8, 21, 100

    /*
    //Versión SFML
    std::cout << SFML_VERSION_MAJOR << "." << SFML_VERSION_MINOR << "." << SFML_VERSION_PATCH << std::endl;
    */

    // Se crea la ventana con (alto, ancho)
    sf::RenderWindow window(sf::VideoMode({ WIDTH, HEIGHT }), "Flow Field C++");
    //window.setFramerateLimit(60); // se limita a 60 FPS
    window.setVerticalSyncEnabled(false); // desactiva VSync

    // Declaración de variables para frame rate en ventana
    sf::Clock fpsClock;
    float updateTimer = 0.f;

    // Se crea un rectángulo que no borra la pantalla por completo, blanco con alpha = 10. Se dibuja encima cada frame,
    // y hace que los trazos viejos de desvanezcan poco a poco
    sf::RectangleShape fadeRect(sf::Vector2f((float)WIDTH, (float)HEIGHT));
    fadeRect.setFillColor(sf::Color(255, 255, 255, 10)); // Estela suave

    // Vector de vectores, en el cual se guardan todos los vectores del flowfield. Cada elemento es un vector2f (x, y),
    // que guarda la dirección que seguirán las partículas
    std::vector<sf::Vector2f> flowfield(flowCount);

    const int N = 5000; //número de partículas: 5000, 10000, 50000, 100000, 500000, 1000000
    std::mt19937 rng(42); //generador de números aleatorios con semilla fija = 42
    std::uniform_real_distribution<float> rx(0.f, (float)WIDTH); //posición x aleatoria
    std::uniform_real_distribution<float> ry(0.f, (float)HEIGHT); //posición y aleatoria
    std::uniform_real_distribution<float> rh(0.f, 360.f); //color aleatorio

    std::vector<Particle> particles;//Se crea el vector de partículas
    particles.reserve(N); //se reserva el espacio con reserve() para optimizar memoria
    for (int i = 0; i < N; ++i) { // Se llena el vector con las partículas, cada una con una posición aleatoria y un color distinto
        particles.emplace_back(rx(rng), ry(rng), rh(rng));
    }

    Perlin3D perlin(seed); //semilla para perlin noise
    float zoff = 0.f; // tiempo -> si se cambia, el campo se mueve; si no -> el flowfield queda fijo

    // Variables para el protocolo de pruebas (warm-up)
    const int WARMUP_FRAMES = 100; //100 frames de calentamiento (se descartan)
    const int TOTAL_TEST_FRAMES = 1000; //frames que sí se miden
    int frameCount = 0; //contador de iteraciones (frames)
    bool warmedUp = false; //bool para indicar si ya se han renderizado los frames de calentamiento

    // Variables para función con GPU
    std::vector<float> xoff_matrix(cols);
    std::vector<float> yoff_matrix(rows);

    for (int i = 0; i < cols; ++i) xoff_matrix[i] = i * inc;
    for (int j = 0; j < rows; ++j) yoff_matrix[j] = j * inc;

    // ============================================================================
    // ----------------------------VALIDACIÓN NUMÉRICA----------------------------
    // Llamada función para validación numérica
    /*
    std::vector<int> n_fotogramas = { 1, 10, 100, 1000, 10000, 100000 };

    for (int valor : n_fotogramas) { //para cada cúmulo de fotogramas
        realizar_test_error(valor, 10000, cols, rows, inc, seed, perlin, xoff_matrix, yoff_matrix);
    }
    */

    // ============================================================================

    //Variables para medir tiempos WARM-UP:
    using clockFPS = std::chrono::high_resolution_clock; //declaración del reloj
    auto startTime = clockFPS::now(); //inicio tiempo
    auto endTime = clockFPS::now(); //fin

    //Variables ms cómputo total en CPU/GPU
    double acumulado_FF = 0.0; // Para sumar los tiempos de cada frame en CPU/GPU -> GENERACIÓN FLOWFIELD
    double acumulado_UpdateParticulas = 0.0; // Para sumar solo la lógica de movimiento -> ACTUALIZACIÓN PARTÍCULAS
    double acumuladoTrans_FF = 0.0;   // Transferencias dentro de launch_cuda_flowfield
    double acumuladoTrans_Part = 0.0; // Transferencias actualización de partículas
    std::chrono::duration<double, std::milli> durationTrans_Inicial = std::chrono::duration<double, std::milli>::zero();

    //Variables para print (Generación del Flowfield / Actualización de partículas) en CPU o GPU
    bool boolFlowfield_cpu = false;
    bool boolParticlesUpdate_cpu = false;

    // ============================================================================
    // 
    // ------------------------ RESERVA DE MEMORIA UNA ÚNICA VEZ PARA ACT. PARTÍCULAS EN GPU ------------------------
    // 

    //SE EJECUTA SIEMPRE SI EL FLOWFIELD SE GENERA EN GPU:
    /*
    float2_simple* d_flowfield = nullptr;
    const size_t FLOW_BYTES = flowCount * sizeof(float2_simple);
    cudaMalloc(&d_flowfield, FLOW_BYTES);

    //COMENTAR SI NO SE HACE EN GPU!!:
    
    // 1. Reservar memoria persistente en la GPU
    ParticleGPU* d_particles = nullptr;
    const size_t PARTICLE_BYTES = N * sizeof(ParticleGPU);

    cudaMalloc(&d_particles, PARTICLE_BYTES);

    // 2. Copiar los datos iniciales (solo una vez)
    auto transfer_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_particles, particles.data(), PARTICLE_BYTES, cudaMemcpyHostToDevice);

    auto transfer_end = std::chrono::high_resolution_clock::now();

    durationTrans_Inicial = transfer_end - transfer_start;
    */

    // ============================================================================

    if (sizeof(Particle) != sizeof(ParticleGPU)) {
        std::cerr << "ERROR: Desajuste de memoria critico!" << std::endl;
        std::cerr << "CPU: " << sizeof(Particle) << " bytes" << std::endl;
        std::cerr << "GPU: " << sizeof(ParticleGPU) << " bytes" << std::endl;
        return -1;
    }

    // loop principal -> mientras la ventana esté activa (abierta)
    while (window.isOpen()) {

        //Frame rate
        float deltaTime = fpsClock.restart().asSeconds();
        updateTimer += deltaTime;

        if (updateTimer >= 0.2f) { // Actualizar solo cada 200ms
            updateTimer = 0.f;
            int fpsValue = static_cast<int>(1.f / (deltaTime + 1e-9f));

            // Construimos el string de forma segura
            std::string fpsStr = "Flow Field C++ | FPS: " + std::to_string(fpsValue);

            // SFML 3.0: Usamos explícitamente u8string o el constructor de sf::String
            // Esto evita que el compilador use punteros temporales inválidos
            window.setTitle(sf::String::fromUtf8(fpsStr.begin(), fpsStr.end()));
        }


        // Cierre de ventana
        std::optional<sf::Event> event;
        while (event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) window.close();
        }

        // Lógica de WARM-UP
        if (!warmedUp && frameCount == WARMUP_FRAMES) {
            std::cout << "Fase de Warm-up terminada. Iniciando medicion real..." << std::endl;
            std::cout << std::endl;
            warmedUp = true;
            startTime = clockFPS::now(); // <--- SE REINICIA EL RELOJ AQUÍ
        }


        // --------- GENERACIÓN DEL FLOWFIELD ---------

        // ============================================================================
        //______________Generación en CPU______________

        
        auto cpu_start = std::chrono::high_resolution_clock::now(); //Inicio

        //1. Llamamos a la función que prepara y lanza el kernel
        flowfield_cpu(perlin, inc, cols, rows, flowfield, zoff);
        boolFlowfield_cpu = true;

        auto cpu_end = std::chrono::high_resolution_clock::now(); //Fin

        // Solo acumulamos si estamos en la fase de medición (tras el warm-up)
        if (warmedUp && frameCount < (WARMUP_FRAMES + TOTAL_TEST_FRAMES)) {
            std::chrono::duration<double, std::milli> frameMS = cpu_end - cpu_start;
            acumulado_FF += frameMS.count();
        }
        
        // ============================================================================



        // ============================================================================
        //______________Generación en GPU______________

        /*
        //1. Llamamos a la función que prepara y lanza el kernel
        GpuMetrics metrics_flowfieldGPU = launch_cuda_flowfield(perlin.p.data(), xoff_matrix.data(), yoff_matrix.data(), zoff, d_flowfield, cols, rows); //h_p, h_xoff, h_yoff, zoff, h_out, cols, rows

        // Solo acumulamos si estamos en la fase de medición (tras el warm-up)
        if (warmedUp && frameCount < (WARMUP_FRAMES + TOTAL_TEST_FRAMES)) {
            // 1. Acumulamos el tiempo de cómputo (Kernel)
            acumulado_FF += metrics_flowfieldGPU.kernelTime;

            // 2. Acumulamos el tiempo de transferencia del Flowfield (que ya se calcula dentro de la función en la GPU)
            acumuladoTrans_FF += metrics_flowfieldGPU.transferTime;

        }
        */
        // ============================================================================



        //2. Aumentamos el tiempo manualmente:
        zoff += 0.001f;



        // ------------- ACTUALIZACIÓN DE PARTÍCULAS Y RENDERIZADO -------------
        // Dibujar estela, haciendo desaparecer lo antiguo poco a poco
        //window.draw(fadeRect);

        // Dibujar partículas
        sf::VertexArray lines(sf::PrimitiveType::Lines); //cada par de vértices, una línea
        // Reservamos memoria para evitar realocaciones constantes (optimización)
        lines.resize(particles.size() * 2); //cada partícula necesita 2 vértices: punto anterior y punto actual
        // por lo que se reserva la memoria una sola vez (que será el número de partículas * 2) para evitar ir ampliando el array

        // ______________1. Actualización de partículas______________

        // ============================================================================
        
        // ______________Actualización en GPU______________
        /*
        // SI EL FLOWFIELD SE GENERA EN CPU:
        // 1. Iniciamos la medida de transferencia antes de la copia del flowfield
        auto trans_step1_start = std::chrono::high_resolution_clock::now();

        // Copiamos el resultado de la CPU a la memoria de la GPU para que el Kernel lo pueda usar
        cudaMemcpy(d_flowfield, flowfield.data(), FLOW_BYTES, cudaMemcpyHostToDevice);

        auto trans_step1_end = std::chrono::high_resolution_clock::now();


        // ------------------

        GpuMetrics metrics_particlesUpdateGPU = launch_cuda_update_particles(d_particles, d_flowfield, N, cols, rows, scl, WIDTH, HEIGHT);
        cudaDeviceSynchronize();

        //Inicio del reloj para medir el tiempo de transferencia de datos entre gpu y cpu
        auto trans_step2_start = std::chrono::high_resolution_clock::now();

        //Transferencia de datos GPU -> CPU
        cudaMemcpy(particles.data(), d_particles, PARTICLE_BYTES, cudaMemcpyDeviceToHost);

        //Fin del reloj
        auto trans_step2_end = std::chrono::high_resolution_clock::now();

        if (warmedUp && frameCount < (WARMUP_FRAMES + TOTAL_TEST_FRAMES)) {
            acumulado_UpdateParticulas += metrics_particlesUpdateGPU.kernelTime; //solo cómputo

            // 2. Tiempo de transferencia de este frame
            std::chrono::duration<double, std::milli> durationTrans_Ida = trans_step1_end - trans_step1_start;
            std::chrono::duration<double, std::milli> durationTrans_Vuelta = trans_step2_end - trans_step2_start;

            acumuladoTrans_Part += (durationTrans_Ida.count() + durationTrans_Vuelta.count());
        }

        // 2. SINCRONIZACIÓN PARA DIBUJAR (CPU)
        // La GPU ya nos devolvió los datos al array 'particles'. Ahora hay que pasarlos al objeto que SFML sabe dibujar.
        for (std::size_t i = 0; i < particles.size(); ++i) {
            auto& p = particles[i];
            std::size_t vIndex = i * 2;

            float dx = std::abs(p.pos.x - p.prevPos.x);
            float dy = std::abs(p.pos.y - p.prevPos.y);

            sf::Color c = sf::Color(234, 137, 154, 25);

            // Si la distancia es muy grande, es un salto de borde. 
            // Dibujamos un punto en lugar de una línea cruzada.
            if (dx > 50.f || dy > 50.f) {
                lines[vIndex].position = p.pos;
            }
            else {
                lines[vIndex].position = p.prevPos;
            }

            lines[vIndex + 1].position = p.pos;
            lines[vIndex].color = c;
            lines[vIndex + 1].color = c;

            p.updatePrev(); // Importante mantener esto para el siguiente ciclo
        }
        */
        // ============================================================================



        // ============================================================================
        // ______________Actualización en CPU______________
        
        boolParticlesUpdate_cpu = true;

        /*
        // 1. Medir transferencia por separado
        // SE PASAN LOS DATOS DEL FLOWFIELD PROCESADOS POR LA GPU A LA VARIABLE FLOWFIELD SI EL FLOWFIELD HA SIDO GENERADO EN GPU
        auto trans_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(flowfield.data(), d_flowfield, FLOW_BYTES, cudaMemcpyDeviceToHost);
        auto trans_end = std::chrono::high_resolution_clock::now();
        */

        // 2. Medir cómputo puro
        auto update_start = std::chrono::high_resolution_clock::now();
        update_particles_cpu(particles, flowfield, lines, cols, rows, scl, WIDTH, HEIGHT);
        auto update_end = std::chrono::high_resolution_clock::now();

        // Acumular tiempos
        if (warmedUp && frameCount < (WARMUP_FRAMES + TOTAL_TEST_FRAMES)) {
            // Sumamos la transferencia al acumulador de transferencias de partículas
            /*
            std::chrono::duration<double, std::milli> transMS = trans_end - trans_start;
            acumuladoTrans_Part += transMS.count();
            */

            // Sumamos el cómputo al acumulador de cómputo
            std::chrono::duration<double, std::milli> frameMS = update_end - update_start;
            acumulado_UpdateParticulas += frameMS.count();
        }
        
        
        // ============================================================================




        // ______________2. Renderizado______________
        window.draw(fadeRect);// Dibujar estela, haciendo desaparecer lo antiguo poco a poco
        window.draw(lines); //Se dibujan todas las líneas de todas las partículas de golpe (más eficiente que dibujar una a una)
        //window.draw(fpsText); //Se muestra el frame rate
        window.display(); //se muestra el frame por pantalla con lo dibujado en window.draw

        //contador de tiempo
        frameCount++; //se aumenta en uno el contador de iteraciones (frames)

        // Si ya calentamos y además pasaron los 1000 frames de prueba:
        if (warmedUp && (frameCount >= WARMUP_FRAMES + TOTAL_TEST_FRAMES)) {
            endTime = clockFPS::now();
            window.close(); // Cerramos automáticamente para la siguiente repetición
        }
        
    }
    
    // ============================================================================

    // ------------------------ RESERVA DE MEMORIA UNA ÚNICA VEZ PARA ACT. PARTÍCULAS EN GPU ------------------------
    /*
    cudaFree(d_flowfield);
    //COMENTAR SI NO SE HACE EN GPU!!
    
    cudaFree(d_particles);
    */
    // ============================================================================

    std::chrono::duration<double> elapsed = endTime - startTime;
    double totalSeconds = elapsed.count();
    double msPerFrame = (totalSeconds * 1000.0) / TOTAL_TEST_FRAMES;

    // 1. Medias de generación del FLowfield
    double media_FFms = acumulado_FF / TOTAL_TEST_FRAMES; //cómputo
    double media_FF_Trans = acumuladoTrans_FF / TOTAL_TEST_FRAMES; //transferencias

    // 2. Medias de Partículas
    double mediaUpdateMs = acumulado_UpdateParticulas / TOTAL_TEST_FRAMES; //cómputo
    double media_Part_Trans = (acumuladoTrans_Part + durationTrans_Inicial.count()) / TOTAL_TEST_FRAMES; //trnsferencias

    // 3. Tiempos totales:
    double total_FF = media_FFms + media_FF_Trans;
    double total_Part = mediaUpdateMs + media_Part_Trans;
    double tiempoTotalSimulacion = total_FF + total_Part;

    // --- Informe Final en Consola ---
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "DATOS DE LA EJECUCION:" << std::endl;
    std::cout << "- Numero de particulas: " << N << std::endl;
    std::cout << "- Dimensiones Flowfield: " << cols << " x " << rows << std::endl;
    std::cout << "- Seed: " << seed << std::endl;
    std::cout << std::endl;
    std::cout << "RESULTADOS DE LA REPETICION:" << std::endl;
    std::cout << "- Frames medidos: " << TOTAL_TEST_FRAMES << " (tras " << WARMUP_FRAMES << " de warm-up)" << std::endl;
    std::cout << "- Tiempo total: " << totalSeconds << " s" << std::endl;
    std::cout << "- Media (ms/frame): " << msPerFrame << " ms" << std::endl;
    std::cout << "- FPS medios: " << TOTAL_TEST_FRAMES / totalSeconds << " FPS" << std::endl;
    std::cout << std::endl;

    //Operador ternario para decidir si imprimimos CPU o GPU
    std::cout << "TIEMPOS DE COMPUTO (SIN TRANSFERENCIAS):" << std::endl;
    std::cout << "- Generacion Flowfield (" << (boolFlowfield_cpu ? "CPU" : "GPU") << "): " << media_FFms << " ms" << std::endl;
    std::cout << "- Actualizacion Particulas (" << (boolParticlesUpdate_cpu ? "CPU" : "GPU") << "): " << mediaUpdateMs << " ms" << std::endl;
    std::cout << std::endl;

    if ((media_FF_Trans || media_Part_Trans) > 0) {
        std::cout << "TIEMPOS DE TRANSFERENCIAS:" << std::endl;
        std::cout << "- Tiempo transferencias generacion flowfield (CPU <-> GPU): " << media_FF_Trans << " ms" << std::endl;
        std::cout << "- Tiempo transferencias act. particulas (CPU <-> GPU): " << media_Part_Trans << " ms" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "TIEMPOS FINALES" << ((media_FF_Trans || media_Part_Trans) > 0 ? " (CON TRANSFERENCIAS): " : ": ") << std::endl;
    std::cout << "- Generacion Flowfield (" << (boolFlowfield_cpu ? "CPU" : "GPU") << "): " << media_FFms + media_FF_Trans << " ms" << std::endl;
    std::cout << "- Actualizacion Particulas (" << (boolParticlesUpdate_cpu ? "CPU" : "GPU") << "): " << mediaUpdateMs + media_Part_Trans << " ms" << std::endl;
    std::cout << "- Tiempo de generacion flowfield + actualizacion particulas" << ((media_FF_Trans || media_Part_Trans) > 0 ? " (incluyendo transferencias): " : ": ") << tiempoTotalSimulacion << " ms" << std::endl;
    
    std::cout << "--------------------------------------" << std::endl;

    return 0; //fin del programa
}