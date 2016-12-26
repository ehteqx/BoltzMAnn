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
NLearnCycles = 2;           % Number of overall learning cycles
Nexamples = 4;              % Number of different examples to learn from
Nvisible = 3;               % Number of visible neurons
Nhidden = 5;                % Number of hidden neurons
Nensemble = 500;            % Number of different ensemble elements
Ntimespan = 10000;          % Timestep(s) duration of time-mean
LearnCoeff = 0.1;           % Learning coefficient
IsRestricted = false;       % BOOLEAN: Is the machine "Restricted"?
IsSelfish = false;          % BOOLEAN: Is the machine "Selfish"?

% TWEAKABLE PARAMETERS (simulated annealer)
Tstart = 1-eps;               % Starting temperature (must be < 1)
SolSteps = 10000;             % Steps to temperature dacay
TDecay = 0.96;                % Temperature dacay factor
AnnSteps = 10000;             % Decay-steps to thermalization

% PREALLOCATION
W = rand([Nvisible+Nhidden,Nvisible+Nhidden]);      % Weight matrix

% PRECOMPUTATION
FinalTempForWUpd = Tstart*((TDecay)^AnnSteps);      % Final temperature

% EXAMPLES ACQUISITION FROM FILE (examples matrix)
X = importdata('examples.txt');

% For every iteration of the overall learning cycle...
for LrnCounter = 1:NLearnCycles

    % For every element of the weight matrix...
    for iW = 1:(Nvisible+Nhidden)           % Iterate rows -- Must remain sequential (NO PARFOR)
        for jW = 1:(Nvisible+Nhidden)       % Iterate cols -- Must remain sequential (NO PARFOR)

            % FREE EVOLUTION (Dreaming...)
            S = randsample([-1, 1], Nvisible+Nhidden, true);        % Neuron states vector (randomization)
            TmpAvgs = zeros(1,Nensemble);
            SParallelized = zeros(Nensemble, Nvisible+Nhidden);
            Scandidate = zeros(Nensemble, Nvisible+Nhidden);

            parfor ensembleNr = 1:Nensemble     % Ensemble averaging

                SParallelized(ensembleNr,:) = S

                for TDecayStep = 1:AnnSteps     % Stochastic annealing thermalization

                    Temperature = Tstart

                    for SolCounter = 1:SolSteps     % Fixed-temperature evolution

                        ChosenOne = randi(Nvisible+Nhidden)

                        Scandidate(ensembleNr,:) = SParallelized(ensembleNr,:)

                        TmpVector = Scandidate(ensembleNr,:)                    % The technique of vector re-injection...
                        TmpVector(ChosenOne) = (-1)*(TmpVector(ChosenOne))      % is necessary in PARFOR loops due to...
                        Scandidate(ensembleNr,:) = TmpVector                    % variable slicing parallelization

                        DeltaH = ((-1/2)*(Scandidate(ensembleNr,:)*(W*(Scandidate(ensembleNr,:)')))) - ((-1/2)*(SParallelized(ensembleNr,:)*(W*(SParallelized(ensembleNr,:)'))))

                        if DeltaH < 0.0         % If the random modification lowers the Energy...
                            SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)      % the solution is accepted immediately
                        else
                            R = rand            % Else, a random number is generated...
                            if exp(-DeltaH/Temperature) > R         % and the solution accepted according to Boltzmann criterion
                                SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)
                            end
                        end
                    end

                    Temperature = Temperature*TDecay

                end

                % The single-system in the ensemble is now thermalized

                for AvgCounter = 1:Ntimespan

                    % Since temperature is now fixed, this is again a fixed-temperature evolution
                    ChosenOne = randi(Nvisible+Nhidden)

                    Scandidate(ensembleNr,:) = SParallelized(ensembleNr,:)

                    TmpVector = Scandidate(ensembleNr,:)                    % The technique of vector re-injection...
                    TmpVector(ChosenOne) = (-1)*(TmpVector(ChosenOne))      % is necessary in PARFOR loops due to...
                    Scandidate(ensembleNr,:) = TmpVector                    % variable slicing parallelization

                    DeltaH = ((-1/2)*(Scandidate(ensembleNr,:)*(W*(Scandidate(ensembleNr,:)')))) - ((-1/2)*(SParallelized(ensembleNr,:)*(W*(SParallelized(ensembleNr,:)'))))

                    if DeltaH < 0.0         % If the random modification lowers the Energy...
                        SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)      % the solution is accepted immediately
                    else
                        R = rand            % Else, a random number is generated...
                        if exp(-DeltaH/Temperature) > R         % and the solution accepted according to Boltzmann criterion
                            SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)
                        end
                    end

                    % And each term of single-system time-averaging <Si*Sj> is summed to the adequate vector
                    PrlvVector = SParallelized(ensembleNr,:)                                            % Again, this is a vector injection...
                    TmpAvgs(1,ensembleNr) = TmpAvgs(1,ensembleNr) + PrlvVector(iW)*PrlvVector(jW)       % required by PARFOR loop

                end

            end

            % Then, the time-average is computed for every single-system in the ensemble
            TmpAvgs = TmpAvgs/Ntimespan;

            % Lastly, ensemble-averaging is computed
            SiSj_free = mean(TmpAvgs);


            % CLAMPED EVOLUTION (Daytime...)
            % Example-specific ensemble-average vector (dummy)
            ExmplVecD = zeros(1,Nexamples);

            % For every example in the examples matrix...
            for ExmCounter = 1:Nexamples

                % The given example is selected
                VisibleExample = X(ExmCounter,:);       % Note that X is already allocated

                S = zeros(1,Nhidden+Nvisible);           % Preallocation
                S(1:Nvisible) = VisibleExample;
                S((Nvisible+1):Nhidden+Nvisible) = randsample([-1, 1], Nhidden, true);      % Acquire visible example and fill with random

                % Now continues almost as in the free case
                TmpAvgs = zeros(1,Nensemble);
                SParallelized = zeros(Nensemble, Nvisible+Nhidden);
                Scandidate = zeros(Nensemble, Nvisible+Nhidden);

                parfor ensembleNr = 1:Nensemble     % Ensemble averaging

                    SParallelized(ensembleNr,:) = S

                    for TDecayStep = 1:AnnSteps     % Stochastic annealing thermalization

                        Temperature = Tstart

                        for SolCounter = 1:SolSteps     % Fixed-temperature evolution

                            ChosenOne = randi(Nhidden)

                            Scandidate(ensembleNr,:) = SParallelized(ensembleNr,:)

                            TmpVector = Scandidate(ensembleNr,:)                    % The technique of vector re-injection...
                            TmpVector(Nvisible+ChosenOne) = (-1)*(TmpVector(Nvisible+ChosenOne))      % is necessary in PARFOR loops due to...
                            Scandidate(ensembleNr,:) = TmpVector                    % variable slicing parallelization

                            DeltaH = ((-1/2)*(Scandidate(ensembleNr,:)*(W*(Scandidate(ensembleNr,:)')))) - ((-1/2)*(SParallelized(ensembleNr,:)*(W*(SParallelized(ensembleNr,:)'))))

                            if DeltaH < 0.0         % If the random modification lowers the Energy...
                                SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)      % the solution is accepted immediately
                            else
                                R = rand            % Else, a random number is generated...
                                if exp(-DeltaH/Temperature) > R         % and the solution accepted according to Boltzmann criterion
                                    SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)
                                end
                            end
                        end

                        Temperature = Temperature*TDecay

                    end

                    % The single-system in the ensemble is now thermalized

                    for AvgCounter = 1:Ntimespan

                        % Since temperature is now fixed, this is again a fixed-temperature evolution
                        ChosenOne = randi(Nhidden)

                        Scandidate(ensembleNr,:) = SParallelized(ensembleNr,:)

                        TmpVector = Scandidate(ensembleNr,:)                    % The technique of vector re-injection...
                        TmpVector(Nvisible+ChosenOne) = (-1)*(TmpVector(Nvisible+ChosenOne))      % is necessary in PARFOR loops due to...
                        Scandidate(ensembleNr,:) = TmpVector                    % variable slicing parallelization

                        DeltaH = ((-1/2)*(Scandidate(ensembleNr,:)*(W*(Scandidate(ensembleNr,:)')))) - ((-1/2)*(SParallelized(ensembleNr,:)*(W*(SParallelized(ensembleNr,:)'))))

                        if DeltaH < 0.0         % If the random modification lowers the Energy...
                            SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)      % the solution is accepted immediately
                        else
                            R = rand            % Else, a random number is generated...
                            if exp(-DeltaH/Temperature) > R         % and the solution accepted according to Boltzmann criterion
                                SParallelized(ensembleNr,:) = Scandidate(ensembleNr,:)
                            end
                        end

                        % And each term of single-system time-averaging <Si*Sj> is summed to the adequate vector
                        PrlvVector = SParallelized(ensembleNr,:)                                            % Again, this is a vector injection...
                        TmpAvgs(1,ensembleNr) = TmpAvgs(1,ensembleNr) + PrlvVector(iW)*PrlvVector(jW)       % required by PARFOR loop

                    end

                end

                % Then, the time-average is computed for every single-system in the ensemble
                TmpAvgs = TmpAvgs/Ntimespan;
                ExmplVecD(ExmCounter) = mean(TmpAvgs);

            end

            SiSj_clamped = mean(ExmplVecD);

            % WEIGHT UPDATE (Remember: temperature is "FinalTempForWUpd")
            DeltaWij = (LearnCoeff/(2*FinalTempForWUpd))*(SiSj_clamped - SiSj_free);

        end
    end

end
