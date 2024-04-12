
class TraceNoise:

    def __init__(self):
        self.noise = list()

    def add_noise(self, noise_type, affected):
        self.noise.append((noise_type, affected))
