% BoltzMAnn - A simulated-annealing-powered (Restricted) Boltzmann Machine
% Copyright (c) 2016 Emanuele Ballarin <emanuele@ballarin.cc>
% Software released under the terms of the MIT License

% The purpose of the script is to implement a computationally-efficient and
% straightforward (optionally Restricted) Boltzmann Machine, powered under the
% hood by Simulated Annealing as a means to 'thermalize' the network.
% The network has been developed to be as general-purpose as possible and the
% choice of the MATLAB language was determined as the optimal compromise between
% runtime performance, conciseness, expressiveness and high-level clarity.

% TWEAKABLE PARAMETERS (whole network)
Nexamples = 1;          % Number of different examples to learn from
Nvisible = 1;           % Number of visible neurons
Nhidden = 1;            % Number of hidden neurons
Nensemble = 1;          % Number of different ensemble elements
Ntimespan = 1;          % Timestep(s) duration of time-mean
LearnCoeff = 1;         % Learning coefficient
IsRestricted = false;   % BOOLEAN: Is the machine "Restricted"?
IsSelfish = false;      % BOOLEAN: Is the machine "Selfish"?

% TWEAKABLE PARAMETERS (simulated annealer)
Tstart = 1;             % Starting temperature (must be < 1)
SolSteps = 1;           % Steps to temperature dacay
TDecay = 1;             % Temperature dacay factor
AnnSteps = 1;           % Decay-steps to thermalization

% PREALLOCATION
W = rand([Nvisible+Nhidden,Nvisible+Nhidden]);       % Weight matrix

% For every element of the weight matrix...
for iW = 1:(Nvisible+Nhidden)     % Iterate rows -- Must remain sequential (NO PARFOR)
    for jW = 1:(Nvisible+Nhidden) % Iterate cols -- Must remain sequential (NO PARFOR)
        
        % FREE EVOLUTION (Dreaming...)
        S = randsample([-1, 1], Nvisible+Nhidden, true);     % Neuron states vector (preallocation)
        EnsembleMeans = zeros(1,Nensemble)
        SParallelized = zeroes(Nensemble, Nvisible+Nhidden)
        Scandidate = zeroes(Nensemble, Nvisible+Nhidden)
        
        parfor ensembleNr = 1:Nensemble     % Ensemble averaging
            
            SParallelized(ensembleNr,:) = S
            
            for TDecayStep = 1:AnnSteps     % Stochastic annealing thermalization
                
                Temperature = Tstart
                
                for SolCounter = 1:SolSteps % Fixed-temperature evolution
                    
                    ChosenOne = randi(Nvisible+Nhidden)
                    
                    Scandidate(ensembleNr,:) = SParallelized(ensembleNr,:)
                    
                    TmpVector = Scandidate(ensembleNr,:)                 % The technique of vector re-injection...
                    TmpVector(ChosenOne) = (-1)*(TmpVector(ChosenOne))   % is necessary in PARFOR loops due to...
                    Scandidate(ensembleNr,:) = TmpVector                 % variable slicing concurrency
                    
                    DeltaH = ((-1/2)*(Scandidate(ensembleNr,:)*(W*(Scandidate(ensembleNr,:)')))) - ((-1/2)*(SParallelized(ensembleNr,:)*(W*(SParallelized(ensembleNr,:)'))))
                    
                    if DeltaH < 0.0
                        SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)
                    else
                        R = rand
                        if exp(-DeltaH/Temperature) > R
                            SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)
                        end                       
                    end                    
                end
                
                Temperature = Temperature*TDecay
                
            end
            
            % The system is now thermalized (sequential averaging)
            
            
                
        end
        
    end    
end
