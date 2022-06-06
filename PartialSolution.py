
def x_polate(known_vals, t, i=0, extrapolate_left=False, extrapolate_right=False):
    if len(known_vals) == 0:
        raise ValueError
    if len(known_vals) == 1:
        return known_vals[0][1], i
    if i >= len(known_vals) - 1:
        i = len(known_vals) - 2
    if i < 0:
        i = 0
    while known_vals[i+1][0] <= t:
        i += 1
        if i == len(known_vals) - 1:
            if extrapolate_right:
                i -= 1
                break
            else:
                return known_vals[-1][1], i
    while known_vals[i][0] > t:
        i -= 1
        if i == -1:
            if extrapolate_left:
                i += 1
                break
            else:
                return known_vals[0][1], i
    diff_s = known_vals[i+1][1] - known_vals[i][1]
    diff_t = known_vals[i+1][0] - known_vals[i][0]
    t -= known_vals[i][0]
    return t / diff_t * diff_s + known_vals[i][1], i

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
        retval, self.i = x_polate(self.known_vals, t, self.i, self.extrapolate_left, self.extrapolate_right)
        return retval