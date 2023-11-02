from numpy import ceil, arange, ndarray
from scipy.integrate import solve_ivp

def SEIRmodel(R0, T, tau) -> ndarray:
    """Return SEIR model solution vector for tsteps in range(0, 500, 0.01)
    
    Args:
        R0: Reproduction number.
        T: Incubation period.
        tau: Infectious period.

    Returns:
        ODE solutions ndarray of shape [len(tsteps), 4] where 
        `S = solutions[:, 0]`, `E = solutions[:, 1]`, `I = solutions[:, 2]`, 
        and `R = solutions[:, 3]`
    """
    N = 1e5 # population size
    b = R0/tau

    # init conditions
    E = 0
    I = 10
    R = 0
    S= N - E - I - R

    # create tsteps at which the solution will be saved
    t0 = 0
    Tend=500
    dt=0.01
    tsteps_to_eval = arange(t0, Tend, dt)

    # ixs 0  1  2  3
    x0 = [S, E, I, R]

    sol = solve_ivp(
        deriv, 
        t_span=(t0, Tend), 
        t_eval=tsteps_to_eval,
        y0=x0, 
        # using RK2 bc that's what MATLAB script uses
        method='RK23',
        args=(b, T, tau))
    
    # [n_state_vars, n_tsteps] --> [n_tsteps, n_state_vars]
    Xoutput = sol.y.T

    return Xoutput


def deriv(t, x, b, T, tau):
    S, E, I, R = x
    N = S + E + I + R

    Sd = -b*S*I/N
    Ed = b*S*I/N - (1/T)*E
    Id = (1/T)*E - (1/tau)*I
    Rd = (1/tau)*I

    return [Sd, Ed, Id, Rd]
