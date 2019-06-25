function asymmetricDiffusionMap = buildAsymmetricDiffusionMap(dynamicsTrace, numNeighbors, localTraceSize, minTime, verbose)
% buildAsymmetricDiffusionMap : This function takes a phase space
% representation of a multivariate time series datastream and converts it
% to a asymmetric diffusion map (transition probabilitiy matrix).
%
% INPUT :
%       dynamicsTrace : n by t time series data where n are the number of
%           dimensions and t are the temporally sequenced observations.
%           This data is assumed to be a true phase space (no trajectory
%           crossing)
%       numNeighbors (optional) : Positive integer (Default 20). Number of
%       trajectories to link in the asymmetric diffusion map.
%       localTraceSize (optional) : Positive integer (Default 4). Number of time
%       steps to include in the local temporal neighborhood for noise
%       estimation.
%       minTime (optional) : Positive integer (Default 10). Minimum number of time
%       steps the trace must take before it can return to the same local
%       neighborhood.
%       verbose (optional) : Logical (Default 0). Toggles drawing of intermediate steps
%
% OUTPUT :
%       asymmetricDiffusionMap : (t-1) by (t-1) matrix representing the
%       probability that the system starting at time t will diffuse to
%       other states at time t+1. The matrix is sparse.
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

%% Default options
if ~exist('verbose') || isempty(verbose)
    DISPLAY = 0;
else
    DISPLAY = verbose;
end

%Preprocessing options
if ~exist('localTraceSize') || isempty(localTraceSize) %Size in time of the local neighbor about each point
    LOCAL_TRACE_SIZE = 4;
else
    LOCAL_TRACE_SIZE = localTraceSize;
end
if ~exist('numNeighbors') || isempty(numNeighbors) %Number of neighbors to add to local neighborhood
    NUM_NEIGHBORS = 20;
else
    NUM_NEIGHBORS = numNeighbors;
end
if ~exist('minTime') || isempty(minTime) %Should the data be detrended?
    MIN_TIME = 10;
else
    MIN_TIME = minTime;
end

%%
stateSpace = dynamicsTrace;

numPoints = size(stateSpace,2) - 1;

localDriftVelocity = [];
normalDriftVelocity = [];
localNoise = [];
localPositionNoise = [];
velocities = diff(stateSpace, 1, 2);

for i = 1:numPoints
    startPoint = max(1, i-LOCAL_TRACE_SIZE);
    endPoint = min(size(velocities,2), i+LOCAL_TRACE_SIZE);
    
    meanPosition = mean(stateSpace(:,startPoint:endPoint), 2);
    centeredPosition = stateSpace(:,startPoint:endPoint) - repmat(meanPosition, 1, length(startPoint:endPoint));
    
    meanVelocity = mean(velocities(:,startPoint:endPoint), 2);
    centeredVelocites = velocities(:,startPoint:endPoint) - repmat(meanVelocity, 1, length(startPoint:endPoint));
    
    localPositionNoise(:,i) = std(centeredPosition, 1, 2);
    localDriftVelocity(:,i) = meanVelocity;
    localNoise(:,i) = std(centeredVelocites, 1, 2);
    normalDriftVelocity(:,i) = meanVelocity / norm(meanVelocity);
end

noiseLevels = sum(localNoise.^2, 1).^0.5;
driftLevels = sum(localDriftVelocity.^2, 1).^0.5;
driftVelocityRatio = driftLevels ./ (noiseLevels / sqrt(2));

sigma = LOCAL_TRACE_SIZE*2;
x = floor(-3*sigma):floor(3*sigma);
filterSize = numel(x);
gaussian = exp(-x.^2/(2*sigma^2))/(2*sigma^2*pi)^0.5;

filteredTrace = driftVelocityRatio;
filteredTrace = cat(2, repmat(filteredTrace(:,1)*0,[1,filterSize]), filteredTrace, repmat(filteredTrace(:,end)*0,[1,filterSize]));
filteredTrace = convn(filteredTrace, gaussian, 'same');
driftVelocityRatio = filteredTrace(:,filterSize+1:end-filterSize);

rootNoise = localNoise.^0.5;
rootDriftRatio = driftVelocityRatio.^0.5;

distances = [];
nearNeighbors = [];
if DISPLAY
    waitHandle = parfor_progressbar(numPoints, 'Calculating nearest neighbors');
else
    disp('Calculating nearest neighbors');
end
parfor i = 1:numPoints
    testPoint = stateSpace(:,i) + localDriftVelocity(:,i);
    
    differences = stateSpace(:,1:numPoints) - repmat(stateSpace(:,i+1), 1, numPoints);
    
    if i < size(localNoise,2)
        differences = differences ./ localNoise ./ repmat(localNoise(:,i+1), 1, size(localNoise,2));
    else
        differences = differences ./ localNoise ./ repmat(localNoise(:,i), 1, size(localNoise,2));
    end
    distance = sum(differences.^2,1);
    
    [~, order] = sort(distance);
    
    maxTrace = min(numPoints, i + 1);
    minTrace = max(1, i + 1);
    thisIndices = minTrace:maxTrace;
    
    if NUM_NEIGHBORS > 1
        order(ismember(order, thisIndices)) = [];
        
        j = 1;
        while j <= length(order)
            difference = abs(order - order(j));
            order(difference > 0 & difference < MIN_TIME) = [];
            
            if j > NUM_NEIGHBORS
                break;
            end
            
            j = j+1;
        end
        
        thisIndices = [thisIndices, order(1:min(NUM_NEIGHBORS, length(order)))];
    end
    
    paddedIndices = [thisIndices, zeros(1, NUM_NEIGHBORS+1 - length(thisIndices))];
    nearNeighbors(:, i) = paddedIndices;
    paddedDistances = distance(thisIndices);
    paddedDistances = [paddedDistances, nan*zeros(1, NUM_NEIGHBORS+1 - length(thisIndices))];
    distances(:, i) = paddedDistances;
    
    if DISPLAY
        waitHandle.iterate(1);
    end
end
if DISPLAY
    close(waitHandle);
end

distances(isinf(distances)) = 0;

closestTrajectory = nanmean(nanmean(distances,1));

weights = sparse(numPoints, numPoints);
if DISPLAY
    waitHandle = parfor_progressbar(numPoints, 'Calculating weights');
end
for i = 1:numPoints
    for j = 1:size(nearNeighbors,1)
        if nearNeighbors(j, i) > 0
            weights(i, nearNeighbors(j, i)) = exp(-distances(j,i)/(2*closestTrajectory));
        end
    end
    
    if DISPLAY
        waitHandle.iterate(1);
    end
end
if DISPLAY
    close(waitHandle);
end

weightType = weights;
weightType(isnan(weightType)) = 0;
zeroIndices = find(sum(weightType, 1)==0);
weightType(zeroIndices,zeroIndices) = 1e-10;
zeroIndices = find(sum(weightType, 2)==0);
weightType(zeroIndices,zeroIndices) = 1e-10;

normalizeVector = full(sum(weightType, 1));

sigma = LOCAL_TRACE_SIZE*4;
x = floor(-3*sigma):floor(3*sigma);
filterSize = numel(x);
gaussian = exp(-x.^2/(2*sigma^2))/(2*sigma^2*pi)^0.5;

filteredTrace = normalizeVector;
filteredTrace = cat(2, repmat(filteredTrace(:,1)*0,[1,filterSize]), filteredTrace, repmat(filteredTrace(:,end)*0,[1,filterSize]));
filteredTrace = convn(filteredTrace, gaussian, 'same');
normalizeVector = filteredTrace(:,filterSize+1:end-filterSize);

normalizeMatrix = (normalizeVector'*normalizeVector).^0.5;
normalizedWeights = weightType ./ normalizeMatrix;
clear 'normalizeMatrix';

probabilityMatrix = normalizedWeights ./ repmat(sum(normalizedWeights, 2), 1, size(normalizedWeights,2));


asymmetricDiffusionMap = probabilityMatrix;

