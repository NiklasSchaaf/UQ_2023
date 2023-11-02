function Xoutput = SEIRmodel(R0,T,tau)

% SEIR model

% parameters:
% R0: reproduction number
% T: length of incubation period
% tau: duration of infectious period

N=1e5; % population size

b=R0/tau;

% initial state:
E=0;
I=10;
R=0;
S=N-E-I-R;

Tend=500;
dt=0.01;
Nt=ceil(Tend/dt);
Xoutput=zeros(Nt,4);

x=[S E I R];

for n=1:Nt
     
    Xoutput(n,:)=x;
    
    % RK2 integration step:
    
    xdot=deriv(x,b,T,tau);
    x2=x+dt*xdot/2;
    xdot2=deriv(x2,b,T,tau);
    x=x+dt*xdot2;
    
end

end

%----------------------------------------------------

function xdot=deriv(x,b,T,tau)

S=x(1); E=x(2); I=x(3); R=x(4);
N=S+E+I+R;

Sd = -b*S*I/N;
Ed = b*S*I/N - (1/T)*E;
Id = (1/T)*E - (1/tau)*I;
Rd = (1/tau)*I;

xdot=[Sd Ed Id Rd];

end




