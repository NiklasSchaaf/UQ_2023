from numpy import ceil, zeros, array

def SEIRmodel(R0,T,tau):
    N = 1e5 # population size
    b= R0/tau

    # init conditions
    E=0
    I=10
    R=0
    S=N-E-I-R

    Tend=500
    dt=0.01
    Nt = int(ceil(Tend/dt))
    Xoutput = zeros(shape=(Nt+1, 4))

    # ixs      0  1  2  3
    x = array([S, E, I, R])

    for n in range(1, Nt+1):
        
        Xoutput[n, :] = x
        
        # RK 2
        xdot = deriv(x,b,T,tau)
        x2 = x + dt*xdot/2
        xdot2 = deriv(x2,b,T,tau)
        x = x + dt*xdot2
        
    return Xoutput


def deriv(x, b, T, tau):
    S, E, I, R = x
    N = S + E + I + R

    Sd = -b*S*I/N
    Ed = b*S*I/N - (1/T)*E
    Id = (1/T)*E - (1/tau)*I
    Rd = (1/tau)*I

    return array([Sd, Ed, Id, Rd])
