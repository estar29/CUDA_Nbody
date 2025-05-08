// Evan Stark - April 28th, 2025 - ITCS 4145 001
// This program will perform the N-Body particle simulation on
// CUDA to speedup that can be achieved on GPUs.

/*
SOURCES USED:
https://www.youtube.com/playlist?list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe (Playlist going over the basics of CUDA and GPU programming).
https://en.cppreference.com/w/cpp/language/constructor (Constructor and member initializer list documentation).
https://en.cppreference.com/w/cpp/numeric/random/random_device (Random device documentation).
https://cplusplus.com/reference/random/mt19937/ (Mersenne-Twister (mt19937) documentation).
https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution (Uniform distribution).
https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution (Normal distribution).
https://cplusplus.com/reference/fstream/fstream/ (fstream library documentation).
https://cplusplus.com/reference/fstream/ifstream/ (ifstream documentation).
https://cplusplus.com/reference/chrono/high_resolution_clock/ (chrono's high resolution clock).
https://cplusplus.com/reference/chrono/duration/ (chrono's duration object).
https://cplusplus.com/reference/chrono/duration_cast/ (duration cast documentation).
https://cplusplus.com/reference/chrono/milliseconds/ (chrono milliseconds).
Majority of code based off of program originally by Erik Saule.
*/

// All needed libraries.
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>

// Gravity constant.
double GRAV = -6.674 * std::pow(10, -11);

struct simulation
{
    size_t num_particles;

    std::vector<double> mass;

    // Vector for forces, velocity, and position and for each direction (x, y, z).
    std::vector<double> pos_x;
    std::vector<double> pos_y;
    std::vector<double> pos_z;

    std::vector<double> force_x;
    std::vector<double> force_y;
    std::vector<double> force_z;

    std::vector<double> vel_x;
    std::vector<double> vel_y;
    std::vector<double> vel_z;

    // Constructor function to make a new simulation object.
    simulation(size_t size):
        num_particles(size), mass(size), pos_x(size), pos_y(size), pos_z(size),
        force_x(size), force_y(size), force_z(size), vel_x(size), vel_y(size), vel_z(size) 
    {}
};

// Setting random values for each particle in a simulation.
void randomize(simulation& s)
{
    // Creating a random RNG device that uses the Mersenne-Twister engine.
    std::random_device ran;
    std::mt19937 mt(ran());

    // Distributing random values for mass, position, and velocity.
    std::uniform_real_distribution<> mass_dis(0.9, 1.0);  // B/w 0.9 and 1.0 (exclusive)
    std::normal_distribution<> pos_dis(0., 1.);
    std::normal_distribution<> vel_dis(0., 1.);

    // For each particle in the simulation, randomly generate values for it.
    for (size_t i = 0; i < s.num_particles; i++)
    {
        // Random mass
        s.mass[i] = mass_dis(mt);
        // Random positions
        s.pos_x[i]=  pos_dis(mt);
        s.pos_y[i] = pos_dis(mt);
        s.pos_z[i] = pos_dis(mt);
        // Random velocities.
        s.vel_x[i] = vel_dis(mt);
        s.vel_y[i] = vel_dis(mt);
        s.vel_z[i] = vel_dis(mt);
    }

}

// Initializing a simulation with solar system planets.
void init_solar(simulation &s)
{
    // Initialize planets and create a simulation object for them.
    enum planets{SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
    s = simulation(10);

    // Setting masses for each planet (in kilograms).
    s.mass[SUN] = 1.9891 * std::pow(10, 30);
    s.mass[MERCURY] = 3.2850 * std::pow(10, 23);
    s.mass[VENUS] = 4.8670 * std::pow(10, 24);
    s.mass[EARTH] = 5.9720 * std::pow(10, 24);
    s.mass[MARS] = 6.3900 * std::pow(10, 23);
    s.mass[JUPITER] = 1.8980 * std::pow(10, 27);
    s.mass[SATURN] = 5.6830 * std::pow(10, 26);
    s.mass[URANUS] = 8.6810 * std::pow(10, 25);
    s.mass[NEPTUNE] = 1.0240 * std::pow(10, 26);
    s.mass[MOON] = 7.3420 * std::pow(10, 22);

    // Using astronomical units to convert positions to meters, velocities to meters per second.
    double au = 1.4960 * std::pow(10, 11);

    s.pos_x = {0, 0.39*au, 0.72*au, 1.00*au, 1.52*au, 5.20*au, 9.58*au, 19.22*au, 30.05*au, au+(3.8440*std::pow(10,8))};
    s.pos_y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    s.pos_z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    s.vel_x = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    s.vel_y = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 30802};
    s.vel_z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

// Resetting all forces before running calculations on them.
void reset_force(simulation &s)
{
    for (size_t i=0; i<s.num_particles; i++)
    {
        s.force_x[i] = 0.;
        s.force_y[i] = 0.;
        s.force_z[i] = 0.;
    }
}

__global__ void update_forces(simulation &s, std::vector<std::vector<double>> &d)
{
    size_t index = threadIdx.x + (blockIdx.x * blockDim.x);

    double soft_factor = 0.01;
    for (size_t i = 0; i < s.num_particles; i++) 
    {
        if (i != index)
        {
            double dist_sq = std::pow(s.pos_x[i]-s.pos_x[index], 2) + std::pow(s.pos_y[i] - s.pos_x[index], 2)
                + std::pow(s.pos_z[i] - s.pos_z[index], 2);
            double FORCE = GRAV * (s.mass[i] - s.mass[index]) / (dist_sq + soft_factor);

            double dir_x = s.pos_x[i] - s.pos_x[index];
            double dir_y = s.pos_y[i] - s.pos_y[index];
            double dir_z = s.pos_z[i] - s.pos_y[index];

            s.force_x[index] += dir_x * FORCE;
            s.force_y[index] += dir_y * FORCE;
            s.force_z[index] += dir_z * FORCE;
        }
    }
}

__global__ void apply_forces(simulation &s, std::vector<std::vector<double>> &d, double time_step)
{
    size_t index = threadIdx.x + (blockIdx.x * blockDim.x);

    s.vel_x[index] += time_step * (s.mass[index] / s.force_x[index]);
    s.vel_y[index] += time_step * (s.mass[index] / s.force_y[index]);
    s.vel_z[index] += time_step * (s.mass[index] / s.force_z[index]);

    s.pos_x[index] += time_step * s.vel_x[index];
    s.pos_y[index] += time_step * s.vel_y[index];
    s.pos_z[index] += time_step * s.vel_z[index];
}

// When it's time to dump the state of the simulation, call this function to print output.
void dump_state(simulation &s) 
{
    // Start by printing the number of particles.
    std::cout << s.num_particles << '\t';
    // Then, for each particle in the simulation, print all of its attributes.
    for (size_t i = 0; i < s.num_particles; i++)
    {
        std::cout << s.mass[i] << '\t';   // Mass
        std::cout << s.pos_x[i] << '\t' << s.pos_y[i] << '\t' << s.pos_z[i] << '\t';    // Position.
        std::cout << s.vel_x[i] << '\t' << s.vel_y[i] << '\t' << s.vel_z[i] << '\t';    // Velocity.
        std::cout << s.force_x[i] << '\t' << s.force_y[i] << '\t' << s.force_z[i] << '\t';  // Forces.
    }

    // Start the next state by printing a new line.
    std::cout << '\n';
}

// Loading a configuration from a file.
void load_from_file(simulation &s, std::string file)
{
    // Using std::ifstream to get input from a file.
    std::ifstream input (file);
    // Retrieve the number of particles from input.
    size_t num_parts;
    input >> num_parts;

    // Initialize the simulation.
    s = simulation(num_parts);
    for (size_t i = 0; i < s.num_particles; i++)
    {
        input >> s.mass[i];
        input >> s.pos_x[i] >> s.pos_y[i] >> s.pos_z[i];
        input >> s.vel_x[i] >> s.vel_y[i] >> s.vel_z[i];
        input >> s.force_x[i] >> s.force_y[i] >> s.force_z[i];
    }

    // If input is not valid/good, throw an exception.
    if (input.good() == false)
    {
        throw "file failed to load!";
    }

}

// Creating the device variants of all the simulation vectors.
std::vector<std::vector<double>> initialize_device_params (simulation &s)
{
    std::vector<double> *d_mass;

    std::vector<double> *d_pos_x;
    std::vector<double> *d_pos_y;
    std::vector<double> *d_pos_z;
    
    std::vector<double> *d_force_x;
    std::vector<double> *d_force_y;
    std::vector<double> *d_force_z;

    std::vector<double> *d_vel_x;
    std::vector<double> *d_vel_y;
    std::vector<double> *d_vel_z;

    // Allocate all memory for each of the device pointers.
    cudaMalloc((void**)&d_mass, sizeof(s.mass));
    
    cudaMalloc((void**)&d_pos_x, sizeof(s.pos_x));
    cudaMalloc((void**)&d_pos_y, sizeof(s.pos_y));
    cudaMalloc((void**)&d_pos_z, sizeof(s.pos_z));

    cudaMalloc((void**)&d_force_x, sizeof(s.force_x));
    cudaMalloc((void**)&d_force_y, sizeof(s.force_y));
    cudaMalloc((void**)&d_force_z, sizeof(s.force_z));

    cudaMalloc((void**)&d_vel_x, sizeof(s.vel_x));
    cudaMalloc((void**)&d_vel_y, sizeof(s.vel_y));
    cudaMalloc((void**)&d_vel_z, sizeof(s.vel_z));

    std::vector<std::vector<double>> device_params;
    device_params.push_back(*d_mass);
    device_params.push_back(*d_pos_x);
    device_params.push_back(*d_pos_y);
    device_params.push_back(*d_pos_z);
    device_params.push_back(*d_force_x);
    device_params.push_back(*d_force_y);
    device_params.push_back(*d_force_z);
    device_params.push_back(*d_vel_x);
    device_params.push_back(*d_vel_y);
    device_params.push_back(*d_vel_z);

    return device_params;
}

void host_to_device(simulation &s, std::vector<std::vector<double>> d) 
{
    cudaMemcpy((void*)&d.at(0), (const void*)&s.mass, sizeof(s.mass), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(1), (const void*)&s.pos_x, sizeof(s.pos_x), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(2), (const void*)&s.pos_y, sizeof(s.pos_y), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(3), (const void*)&s.pos_z, sizeof(s.pos_z), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(4), (const void*)&s.force_x, sizeof(s.force_x), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(5), (const void*)&s.force_y, sizeof(s.force_y), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(6), (const void*)&s.force_z, sizeof(s.force_z), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(7), (const void*)&s.vel_x, sizeof(s.vel_x), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(8), (const void*)&s.vel_y, sizeof(s.vel_y), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)&d.at(0), (const void*)&s.vel_z, sizeof(s.vel_z), cudaMemcpyHostToDevice);
}


void device_to_host(simulation &s, std::vector<std::vector<double>> d)
{
    cudaMemcpy((void*)&s.mass, (const void*)&d.at(0), sizeof(d.at(0)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.pos_x, (const void*)&d.at(1), sizeof(d.at(1)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.pos_y, (const void*)&d.at(2), sizeof(d.at(2)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.pos_z, (const void*)&d.at(3), sizeof(d.at(3)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.force_x, (const void*)&d.at(4), sizeof(d.at(4)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.force_y, (const void*)&d.at(5), sizeof(d.at(5)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.force_z, (const void*)&d.at(6), sizeof(d.at(6)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.vel_x, (const void*)&d.at(7), sizeof(d.at(7)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.vel_y, (const void*)&d.at(8), sizeof(d.at(8)), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)&s.vel_z, (const void*)&d.at(9), sizeof(d.at(9)), cudaMemcpyDeviceToHost);
}

void free_device(std::vector<std::vector<double>> d)
{
    for (size_t i = 0; i < d.size(); i++)
    {
        cudaFree((void*)&d.at(i));
    }
}

// Main driver function.
int main(int argc, char* argv[])
{
    // If there are not 6 arguments passed in, tell the user what output is allowed and close the function.
    if (argc != 6)
    {
        std::cout << "Please enter in the following parameters:\n"; 
        std::cout << argv[0] << "<input: number of particles or file to pass in> <time step size> <number of time steps> <how often to dump state> <integer size of each CUDA block>";
        return -1;
    }

    // If the parameters are correct, assign the values.
    double step_size = std::atof(argv[2]);
    size_t num_steps = std::atoi(argv[3]);
    size_t dump_rate = std::atoi(argv[4]);
    size_t threads_per_block = std::atoi(argv[5]);

    // First initialize a simulation with just 1 particle (will not be used).
    simulation s(1);

    // Getting the number of particles in the simulation based on the type of argv[1].
    // If argv[1] is an integer, simply create a simulation with argv[1] particles.
    size_t num_parts = std::atoi(argv[1]);
    if (num_parts > 0)
    {
        s = simulation(num_parts);
        randomize(s);
    }

    // If argv[1] is not an integer, try to either load a solar system sim or from a file depending on argv[1]'s string value.
    else
    {
        std::string file = argv[1];
        // If the filename is either solar or planet, load new solar system simulation.
        if (file == "solar" || file == "planet")
        {
            init_solar(s);
        }
        // Else, attempt to load from the filename.
        else
        {
            load_from_file(s, file);
        }
    }

    // Start timing the computation.
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto start_time = high_resolution_clock::now();

    // Creating the device variants of all the simulation vectors.
    std::vector<std::vector<double>> device_params = initialize_device_params(s);

    // Loading host vector data onto the device vectors.
    host_to_device(s, device_params);

    size_t num_blocks = s.num_particles / threads_per_block;

    // Rounding up if a remainder exists.
    if (s.num_particles % threads_per_block != 0)
    {
        num_blocks += 1;
    }


    // Running the simulation.
    // 
    for (size_t step = 0; step < num_steps; step++)
    {
        // If step is a multiple of dump_rate, dump the state.
        // Also dumps the very first state.
        // Additionally, copies the data back from the device back to the host and loads it back up later.
        if ( (step % dump_rate) == 0)
        {
            device_to_host(s, device_params);
            dump_state(s);
            host_to_device(s, device_params);
        }

        // Reset the state before making calculations.
        reset_force(s);

        // For every two pairs of particles, calculate the forces they apply on one another.
        for (size_t i = 0; i < s.num_particles; i++)
        {
            update_forces<<<num_blocks, threads_per_block>>>(s, device_params);
            apply_forces<<<num_blocks, threads_per_block>>>(s, device_params, step_size);
        }

    }
    
    // Cuda free are the device variables.
    free_device(device_params);

    auto end_time = high_resolution_clock::now();

    duration<double, std::milli> total_time = end_time - start_time;

    std::cout << "Computation time: " << total_time.count() << "\n";
    
    // End program.
    return 0;
}
