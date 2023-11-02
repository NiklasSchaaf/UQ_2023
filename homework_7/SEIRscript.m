% this script integrates the SEIR model 20 times
% with newly sampled random parameter for each integration

figure(1); clf;

for n=1:20

    R0=2.2;
    T=9; 
    tau=1.0+13*betarnd(2,2,1,1);
    
    Xoutput = SEIRmodel(R0,T,tau);
    
    Q = max(Xoutput(:,3)); % maximum of timeseries of I
   
    figure(1); subplot(2,1,1);
    plot(Xoutput(:,3),'-r'); hold on; % plot timeseries of I
    xlabel('timesteps'); ylabel('I')
    subplot(2,1,2);
    plot(tau,Q,'or'); hold on;
    xlabel('\tau'); ylabel('Q')
    
end

        
