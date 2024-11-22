import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies

"""

class NBodySimulator:

    def __init__(self, particles: Particles):
        
        # TODO
        self.particles = particles
        self.G = 1
        self.rsoft = 0.01
        self.method = "RK4"
        self.io_freq = 10
        self.io_header = "nbody"
        self.io_screen = True
        self.visualization = False

        return

    def setup(self, G=1,
                    rsoft=0.01,
                    method="RK4",
                    io_freq=10,
                    io_header="nbody",
                    io_screen=True,
                    visualization=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_header: the output header
        :param io_screen: print message on screen or not.
        :param visualization: on the fly visualization or not. 
        """
        
        # TODO
        self.G = G
        self.rsoft = rsoft
        self.method = method
        self.io_freq = io_freq
        self.io_header = io_header
        self.io_screen = io_screen
        self.visualization = visualization

        return

    def evolve(self, dt:float, tmax:float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """

        # TODO
        folder = f"data_{self.io_header}" 
        Path(folder).mkdir(parents=True, exist_ok=True)  
        
        method_map = {
            "Euler": self._advance_particles_Euler,
            "RK2": self._advance_particles_RK2,
            "RK4": self._advance_particles_RK4
        }
        
        advance_method = method_map.get(self.method)
        if advance_method is None:
            raise ValueError(f"Invalid integration method: {self.method}")

        t = 0
        id = 0
        while t < tmax:

            self.particles = advance_method(dt, self.particles)
            fn = f"{folder}/{self.io_header}_{id:06d}.dat"
            self.particles.output(fn)
            if id % self.io_freq == 0:
                
                if self.io_screen:
                    print(f"Time: {t:.2f}/{tmax:.2f}")

                if self.visualization:
                    self.particles.draw(2)

            t += dt
            id += 1

        print("Simulation is done!")
        return
    


    @jit(forceobj=True)
    def _calculate_acceleration(self, nparticles, masses, positions):
        """
        Calculate the acceleration of the particles
        """

        # TODO
        G = self.G
        rsoft = self.rsoft

        # invoke the kernel for acceleration calculation
        accelerations = _calculate_acceleration_kernel(nparticles, masses, positions, G, rsoft)

        return accelerations

    

    @jit(forceobj=True)
    def _advance_particles_Euler(self, dt, particles):

        #TODO
        accelerations = self._calculate_acceleration(particles.nparticles, particles.masses, particles.positions)
        particles.positions += dt * particles.velocities
        particles.velocities += dt * accelerations
        return particles

    @jit(forceobj=True)
    def _advance_particles_RK2(self, dt, particles):

        # TODO
        positions_half = particles.positions + 0.5 * dt * particles.velocities
        accelerations_half = self._calculate_acceleration(particles.nparticles, particles.masses, positions_half)
        particles.positions += dt * particles.velocities
        particles.velocities += dt * accelerations_half
        return particles

    @jit(forceobj=True)
    def _advance_particles_RK4(self, dt, particles):
        
        #TODO
        k1_v = dt * self._calculate_acceleration(particles.nparticles, particles.masses, particles.positions)
        k1_x = dt * particles.velocities

        k2_v = dt * self._calculate_acceleration(particles.nparticles, particles.masses, particles.positions + 0.5 * k1_x)
        k2_x = dt * (particles.velocities + 0.5 * k1_v)

        k3_v = dt * self._calculate_acceleration(particles.nparticles, particles.masses, particles.positions + 0.5 * k2_x)
        k3_x = dt * (particles.velocities + 0.5 * k2_v)

        k4_v = dt * self._calculate_acceleration(particles.nparticles, particles.masses, particles.positions + k3_x)
        k4_x = dt * (particles.velocities + k3_v)

        particles.positions += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        particles.velocities += (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        return particles

@njit(parallel=True)
def _calculate_acceleration_kernel(nparticles, masses, positions, G, rsoft):

    accelerations = np.zeros_like(positions)
    for i in prange(nparticles):
        for j in prange(nparticles):
            if (j>i): 
                rij = positions[i,:] - positions[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                force = - G * masses[i,0] * masses[j,0] * rij / r**3
                accelerations[i,:] += force[:] / masses[i,0]
                accelerations[j,:] -= force[:] / masses[j,0]

    return accelerations


if __name__ == "__main__":
    
    pass
