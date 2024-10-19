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
        if value.shape != (self.nparticles,):
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

    def add_particles(self, masses, positions, velocities, accelerations):
        # Validate shapes of the input arrays
        if masses.shape != (masses.shape[0], 1):
            raise ValueError(f"Masses should be a 2D array with shape (num_particles, 1)")
        if positions.shape != (positions.shape[0], 3):
            raise ValueError(f"Positions should be a 2D array with shape (num_particles, 3)")
        if velocities.shape != (velocities.shape[0], 3):
            raise ValueError(f"Velocities should be a 2D array with shape (num_particles, 3)")
        if accelerations.shape != (accelerations.shape[0], 3):
            raise ValueError(f"Accelerations should be a 2D array with shape (num_particles, 3)")

        # Check if all arrays have the same number of particles
        num_new_particles = masses.shape[0]
        if not (positions.shape[0] == num_new_particles and
                velocities.shape[0] == num_new_particles and
                accelerations.shape[0] == num_new_particles):
            raise ValueError("All input arrays must have the same number of particles")

        # Append the new particles' data to the existing data
        self._masses = np.vstack((self._masses, masses))
        self._positions = np.vstack((self._positions, positions))
        self._velocities = np.vstack((self._velocities, velocities))
        self._accelerations = np.vstack((self._accelerations, accelerations))

        # Update the number of particles
        self.nparticles += num_new_particles


# Create a Particles instance
# num_particles = 100
# particles = Particles(num_particles)

# Uncomment each line to test and ensure it raises an error
# particles.masses = np.ones(num_particles)  # This line will work fine
# particles.positions = np.random.rand(199, 3)  # This line will raise an error
# particles.velocities = np.random.rand(500, 3)  # This line will raise an error
#


