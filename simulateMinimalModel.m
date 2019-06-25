function simulatedActivity = simulateMinimalModel(manifoldMap, manifoldTransitions, manifoldWeights, embeddedTrace, numSimulatedSteps, verbose)
% simulateMinimalModel : This function simulates a minimal model of a multivariate
% time series data stream and converts the simulation bacl to the original
% activity space.
%
% INPUT : 
%       manifoldMap : Manifold space position of observed points computed in
%       previous step
%       manifoldTransitions : Manifold space transition probability matrix
%       from previous step
%       manifoldWeights : Loadings of observations on each manifold bin
%       embeddedTrace : Observed trace. From previous steps
%       numSimulatedSteps (optional) : Positive integer (Default 100000).
%       Number of steps to run the simulation for
%       verbose (optional) : Logical (Default 0). Toggles drawing of intermediate steps    
%           
% OUTPUT : 
%       simulatedActivity : numSimulatedSteps by n matrix of simualted trace
%       activity. Same number of units as original trace
%
% Copyright (C) 2019 Proekt Lab 
% Written by Connor Brennan.
% University of Pennsylvania, 2019
% This file is part of functions for Brennan and Proekt. eLife 2019.
% 
% This script is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This script is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this script.  If not, see <http://www.gnu.org/licenses/>.

%%
if ~exist('verbose') || isempty(verbose)
    DISPLAY = 0;
else
    DISPLAY = verbose;
end

%Preprocessing options
if ~exist('numSimulatedSteps') || isempty(numSimulatedSteps) %Number of times to simulate model
    NUM_SIMULATED_STEPS = 100000; 
else
    NUM_SIMULATED_STEPS = numSimulatedSteps;
end

%%
stateMap = manifoldMap;
transitionProbabilities = manifoldTransitions;

[uniqueStates, stateUniqueMap, stateIDs] = unique(stateMap, 'rows');
stateBackMap = stateMap(stateUniqueMap,:);

steadyState = [];
for i = 1:size(uniqueStates,1)
    steadyState(i) = sum(stateIDs == i);
    
    steadyStateSpace(stateBackMap(i,2), stateBackMap(i,1)) = sum(stateIDs == i);
end

currentState = [];
[~,currentState(1)] = max(steadyState);

for i = 2:NUM_SIMULATED_STEPS
    currentState(i) = randsample(1:size(transitionProbabilities,2), 1, true, transitionProbabilities(currentState(i-1),:));
end

finalDistribution = [];
for i = 1:size(transitionProbabilities,2)
    finalDistribution(i) = sum(currentState == i);
end

finalDistribution = finalDistribution ./ sum(finalDistribution(:));

finalDistributionSpace = [];
for i = 1:size(uniqueStates,1)
    finalDistributionSpace(stateBackMap(i,2), stateBackMap(i,1)) = finalDistribution(i);
end

if DISPLAY
    figure(7);
    clf;
    subplot(1,2,1);
    imagesc(steadyStateSpace);
    title('Observed');
    xlabel('Loop ID');
    ylabel('Phase bin');    
    subplot(1,2,2);
    imagesc(finalDistributionSpace);
    title('Simulated');
    xlabel('Loop ID');
    ylabel('Phase bin');
    c = colorbar;
    c.Label.String = 'State occupancy (au)';
    suptitle('Steady state of simulation');
    
    figure(8);
    clf;
    subplot(2,1,1);
    plot(stateIDs);
    xlabel('Time (timesteps)');
    ylabel('State ID');
    title('Observed');
    subplot(2,1,2);
    plot(currentState(1:length(stateIDs)));
    xlabel('Time (timesteps)');
    ylabel('State ID');
    title('Simulated');
    suptitle('Manifold space simulation');
end

%%

embeddedTrace = embeddedTrace(:,1:size(manifoldWeights,1));

activitySpaceMap = [];
for i = 1:size(uniqueStates,1)
    stateTrajectoryID = stateBackMap(i,:);
    statePhase = stateTrajectoryID(2);
    stateCommunity = stateTrajectoryID(1);
    
    activitySpaceMap(i,:) = embeddedTrace * manifoldWeights(:,statePhase,stateCommunity);
end

simulatedActivity = activitySpaceMap(currentState,:);


