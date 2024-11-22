import numpy as np
import matplotlib.pyplot as plt
 
class Particles:
    """
    Particle class to store particle properties
    """
    def __init__(self, N):
        self.nparticles = N
        self._masses = None
        self._positions = None
        self._velocities = None
        self._accelerations = None
        self._tags = None

    @property
    def masses(self):
        return self._masses

    @masses.setter
    def masses(self, value):
        # Check if the shape of masses array is correct
        if value.shape != np.ones((self.nparticles,1)).shape:
            raise ValueError(f"Masses should be a 1D array of length {self.nparticles}")
        self._masses = value

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        # Check if the shape of positions array is correct
        if value.shape != (self.nparticles, 3):
            raise ValueError(f"Positions should be a 2D array with shape ({self.nparticles}, 3)")
        self._positions = value

    @property
    def velocities(self):
        return self._velocities

    @velocities.setter
    def velocities(self, value):
        # Check if the shape of velocities array is correct
        if value.shape != (self.nparticles, 3):
            raise ValueError(f"Velocities should be a 2D array with shape ({self.nparticles}, 3)")
        self._velocities = value

    @property
    def accelerations(self):
        return self._accelerations

    @accelerations.setter
    def accelerations(self, value):
        # Check if the shape of accelerations array is correct
        if value.shape != (self.nparticles, 3):
            raise ValueError(f"Accelerations should be a 2D array with shape ({self.nparticles}, 3)")
        self._accelerations = value

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        # Check if the shape of tags array is correct
        if value.shape != (self.nparticles,):
            raise ValueError(f"Tags should be a 1D array of length {self.nparticles}")
        self._tags = value

    def add_particles(self, mass, pos, vel, acc):
        self.nparticles += mass.shape[0]
        self.masses = np.vstack((self.masses, mass))
        self.positions = np.vstack((self.positions, pos))
        self.velocities = np.vstack((self.velocities, vel))
        self.accelerations = np.vstack((self.accelerations, acc))
        self.tags = np.arange(self.nparticles)
        return
    
    def compute_kinetic_energy(self):

        # return 0.5 * np.sum(self._masses * np.linalg.norm(self._velocities, axis=1)**2)
        return np.sum((self.velocities[:, 0] ** 2 + self.velocities[:, 1] ** 2 + self.velocities[:, 2] ** 2) \
               * self.masses / 2)
    def compute_potential_energy(self):

        potential_energy = 0
        for i in range(self.nparticles):
            for j in range(i + 1, self.nparticles):
                r_ij = np.linalg.norm(self._positions[i] - self._positions[j])
                potential_energy -= 10 * self._masses[i] * self._masses[j] / r_ij
        return float(potential_energy)

    def output(self, filename):    
        masses_reshaped = self._masses.reshape(-1, 1)
        data = np.hstack((masses_reshaped, self._positions, self._velocities, self._accelerations))
        kinetic_energy = self.compute_kinetic_energy()
        potential_energy = self.compute_potential_energy()
        total_energy = kinetic_energy + potential_energy
        with open(filename, 'w') as f:
            f.write(f'# Total Kinetic Energy: {kinetic_energy} J\n')
            f.write(f'# Total Potential Energy: {potential_energy} J\n')
            f.write(f'# Total Energy: {total_energy} J\n')
            np.savetxt(f, data, header='mass x y z vx vy vz ax ay az', comments='# ')

    def draw(self, dim):
        if dim == 2:
            plt.scatter(self._positions[:, 0], self._positions[:, 1], s=1, alpha = 0.5)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self._positions[:, 0], self._positions[:, 1], self._positions[:, 2])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()
        else:
            raise ValueError("Invalid dimension.")
        



if __name__ == "__main__":
    # Create a Particles instance
    num_particles = 100
    particles = Particles(num_particles)

    # Uncomment each line to test and ensure it raises an error
    particles.masses = np.ones(num_particles)  # This line will work fine
    particles.positions = np.random.rand(199, 3)  # This line will raise an error
    particles.velocities = np.random.rand(500, 3)  # This line will raise an error


# Create a Particles instance
# num_particles = 100
# particles = Particles(num_particles)

# Uncomment each line to test and ensure it raises an error
# particles.masses = np.ones(num_particles)  # This line will work fine
# particles.positions = np.random.rand(199, 3)  # This line will raise an error
# particles.velocities = np.random.rand(500, 3)  # This line will raise an error
#


