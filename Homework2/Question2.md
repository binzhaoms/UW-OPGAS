
def velocity_func(t, A, B, C, D):
    return A * np.cos(B * t) + C * t + D

  # Initial guesses
    p0 = [3, np.pi/4, 2/3, 32]
    p1 = [0, 0, 0, 0]
    p2 = [6, np.pi/4, 2/3, 32]
    p3 = [3, np.pi/2, 2/3, 32]
    p4 = [3, np.pi/4, 4/3, 32]
    p5 = [3, np.pi/4, 2/3, 64]
    
    
    
    The E2 error metric quantitatively confirms what we saw visually: the oscillatory fits (p0, p2, p4, p5) provide much better approximations to the data than the linear-like fits (p1, p3). This demonstrates how different initial guesses can lead to solutions of varying quality in nonlinear least squares problems.

Seems B is playing a more important role when we pick init guess.
Let's add:
    p6= [0, np.pi/4, 0, 0]
    p7 = [100, np.pi/4, 100, 100]
And we still get the same sit. Align with the assumption.

