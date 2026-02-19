#include <SFML/Graphics.hpp> //dibujar ventanas, líneas, colores, maejar eventos, etc.
#include <cstdint>
#include <vector> //guardar listas dinámicas
#include <random> //para posiciones iniciales aleatorias
#include <cmath> //sin, cos, sqrt, floor
#include <algorithm> //shiffle, iota
#include <numeric>

//medir tiempo:
#include <chrono>
#include <iostream>

//kernel.h para kernel
#include "kernel.h"

// ------------------------ Perlin Noise (Improved + Fractal) ------------------------
struct Perlin3D {
    std::vector<int> p;

    Perlin3D(unsigned seed = 1337) {
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
    zoff += 0.0003f;
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

    const int N = 5000; //número de partículas: 100, 1000, 5000, 10000, 20000, 50000
    std::mt19937 rng(42); //generador de números aleatorios con semilla fija = 42
    std::uniform_real_distribution<float> rx(0.f, (float)WIDTH); //posición x aleatoria
    std::uniform_real_distribution<float> ry(0.f, (float)HEIGHT); //posición y aleatoria
    std::uniform_real_distribution<float> rh(0.f, 360.f); //color aleatorio

    std::vector<Particle> particles;//Se crea el vector de partículas
    particles.reserve(N); //se reserva el espacio con reserve() para optimizar memoria
    for (int i = 0; i < N; ++i) { // Se llena el vector con las partículas, cada una con una posición aleatoria y un color distinto
        particles.emplace_back(rx(rng), ry(rng), rh(rng));
    }

    Perlin3D perlin(1337); //semilla para perlin noise
    float zoff = 0.f; // tiempo -> si se cambia, el campo se mueve; si no -> el flowfield queda fijo

    //para medir tiempo:
    const int MAX_FRAMES = 1000; //número máximo de iteraciones (frames)
    int frameCount = 0; //contador de iteraciones (frames)

    using clock = std::chrono::high_resolution_clock; //declaración del reloj
    auto startTime = clock::now(); //inicio tiempo

    // Variables para función con GPU
    std::vector<float> xoff_matrix(cols);
    std::vector<float> yoff_matrix(rows);

    for (int i = 0; i < cols; ++i) xoff_matrix[i] = i * inc;
    for (int j = 0; j < rows; ++j) yoff_matrix[j] = j * inc;


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
        //sf::Event event;
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) window.close();
        }

        // --------- GENERACIÓN DEL FLOWFIELD ---------

        //______________Generación en CPU______________
        //flowfield_cpu(perlin, inc, cols, rows, flowfield, zoff);

        //______________Generación en GPU______________
        //1. Llamamos a la función que prepara y lanza el kernel
        launch_cuda_flowfield(perlin.p.data(), xoff_matrix.data(), yoff_matrix.data(), zoff, flowfield.data(), cols, rows); //h_p, h_xoff, h_yoff, zoff, h_out, cols, rows

        //2. Aumentamos el tiempo manualmente:
        zoff += 0.001f; 

        // ------------- RENDERIZADO -------------
        // Dibujar estela, haciendo desaparecer lo antiguo poco a poco
        window.draw(fadeRect);

        // Dibujar partículas
        sf::VertexArray lines(sf::PrimitiveType::Lines); //cada par de vértices, una línea
        // Reservamos memoria para evitar realocaciones constantes (optimización)
        lines.resize(particles.size() * 2); //cada partícula necesita 2 vértices: punto anterior y punto actual
        // por lo que se reserva la memoria una sola vez (que será el número de partículas * 2) para evitar ir ampliando el array

        //Se recorre cada una de las partículas (i es el índice de la partícula actual):
        for (std::size_t i = 0; i < particles.size(); ++i) {

            auto& p = particles[i]; // Referencia a la partícula actual

            int idx = p.index(cols, rows, scl); //convertimos la posición en píxeles de la partícula en una celda del
            // flowfield (idx es el índice del vector flowfield que le corresponde)
            p.applyForce(flowfield[(std::size_t)idx]); //Se coge el vector del flowfield en esa celda y se aplica como fuerza a la partícula (vector)

            p.update(); //se actualizan las físicas del vector p -> velocidad += aceleración, se limita la velocidad,
            // posición += velocidad, aceleración = 0
            p.edges(WIDTH, HEIGHT); //Si la partícula se sale de la pantalla, reaparece por el lado contrario
            // para evitar que desaparezcan

            sf::Color c = HSVtoRGBA(p.hue, 255, 255, 20); //Convierte el tono (hue) de la partícula en un
            // color real con saturación máxima, brillo máximo y alpha = 20 (muy transparente), creando así líneas suaves
            // efecto de estela y acumulación de color

            // Calculamos el índice en el array de vértices
            std::size_t vIndex = i * 2;

            // Se escribe en el array de vértices la posición anterior de la partícula y la actual, ambas con mismo color
            lines[vIndex] = sf::Vertex({ p.prevPos, c });
            lines[vIndex + 1] = sf::Vertex({ p.pos, c });

            p.updatePrev(); //Se guarda la posición actual como la anterior para preparar para el siguiente frame y que
            // la estela continue correctamente

        }

        window.draw(lines); //Se dibujan todas las líneas de todas las partículas de golpe (más eficiente que
        // dibujar una a una)
        //window.draw(fpsText); //Se muestra el frame rate
        window.display(); //se muestra el frame por pantalla con lo dibujado en window.draw

        //contador de tiempo
        frameCount++; //se aumenta en uno el contador de iteraciones (frames)

        /*
        //Si se ha llegado al número máximo de iteraciones, se cierra la ventana
        if (frameCount >= MAX_FRAMES) {
            window.close();
        }
        */
    }

    auto endTime = clock::now(); //tiempo final
    std::chrono::duration<double> elapsed = endTime - startTime; //tiempo transcurrido

    //Impresión por pantalla datos
    std::cout << "Particles: " << N << "\n";
    std::cout << "Flowfield: " << cols << " x " << rows << "\n";
    std::cout << "Frames: " << frameCount << "\n";
    std::cout << "Total time (s): " << elapsed.count() << "\n";
    std::cout << "Time per frame (ms): " << (elapsed.count() * 1000.0 / frameCount) << "\n\n";


    return 0; //fin del programa
}