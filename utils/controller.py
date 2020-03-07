class EpsilonController():
    def __init__(self, initial_e=1., e_decays=10000, e_min=0.1):
        self._count = 0
        self._init_e = initial_e
        self._e_min = e_min
        self._e_decay = (initial_e - e_min) / e_decays
        self.num_decays = e_decays

        self.e = initial_e

    def val(self):
        """return value of epsilon"""
        return self.e

    def update(self):
        """update value of epsilon"""
        self._count += 1
        self.e = max(self.e - self._e_decay, self._e_min)

        if self._count == self.num_decays/2 or self._count == self.num_decays:
            print("epsilon set to: ", self.e)

    def count(self):
        return self._count

    def reset(self):
        self.e = self._init_e
        self.count = 0
        return self.e
