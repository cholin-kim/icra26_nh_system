from filterpy.monte_carlo import systematic_resample
import numpy as np
# from filterpy.stats import likelihood


class SimpleParticleFilter:
    def __init__(self, num_particles, state_dim):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.normal(0, 0.01, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, noise_std):
        self.particles += np.random.normal(0, noise_std, np.shape(self.particles))

    def update(self, likelihood_func):
        # likelihood_func: 사용자 정의 관측 확률 함수
        self.weights *= likelihood_func(self.particles)
        self.weights += 1.e-300  # avoid zero weight
        self.weights /= sum(self.weights)

        # Resample if necessary
        if 1. / np.sum(self.weights**2) < self.num_particles / 5.0:

            indexes = systematic_resample(self.weights)
            print()
            print('resampled!')
            print()
            self.particles = self.particles[indexes]
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)