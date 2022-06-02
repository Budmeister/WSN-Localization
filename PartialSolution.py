

class PartialSolution:
    def __init__(self, initial_value : tuple = None, extrapolate_left=False, extrapolate_right=False):
        self.known_vals = []
        self.i = 0
        self.extrapolate_left = extrapolate_left
        self.extrapolate_right = extrapolate_right
        if initial_value is not None:
            self.add_val(*initial_value)
    
    def add_val(self, t, s):
        for i in range(len(self.known_vals)):
            if self.known_vals[i][0] > t:
                self.known_vals.insert(i, (t, s))
                break
            elif self.known_vals[i][0] == t:
                self.known_vals[i] = (t, s)
                break
        else:
            self.known_vals.append((t, s))
    
    def __call__(self, t):
        if len(self.known_vals) == 1:
            return self.known_vals[0][1]
        if self.i >= len(self.known_vals) - 1:
            self.i = len(self.known_vals) - 2
        if self.i < 0:
            self.i = 0
        while self.known_vals[self.i+1][0] <= t:
            self.i += 1
            if self.i == len(self.known_vals) - 1:
                if self.extrapolate_right:
                    break
                else:
                    return self.known_vals[-1][1]
        while self.known_vals[self.i][0] > t:
            self.i -= 1
            if self.i == -1:
                if self.extrapolate_left:
                    break
                else:
                    return self.known_vals[0][1]
        diff_s = self.known_vals[self.i+1][1] - self.known_vals[self.i][1]
        diff_t = self.known_vals[self.i+1][0] - self.known_vals[self.i][0]
        t -= self.known_vals[self.i][0]
        return t / diff_t * diff_s + self.known_vals[self.i][1]