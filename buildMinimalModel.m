function [manifoldSpaceMap, manifoldSpaceTransitions, manifoldSpaceWeights] = buildMinimalModel(transitionMatrix, embeddedTrace, stateDensity, minStateTime, gamma, maxCommunityAttempts, phaseCounts, verbose)
% buildMinimalModel : This functions takes a asymmetric diffusion map and
% builds a minimal model of loops and the phase
% along those loops.
%
% INPUT : 
%       transitionMatrix : Transition probability matrix computed in
%       previous step.
%       embeddedTrace : Dynamics trace on which the transition probability
%       matrix was computed (from previous step).
%       stateDensity (optional) : Number between 0 and 1 (Default 0.25).
%       The minimum density of states in the transition probability matrix.
%       Ensures that spectral analysis doesn't error out.
%       minStateTime (optional) : Positive integer (Default 5). Minimum
%       time a state can last. If less than this then merge the state into
%       border states.
%       gamma (optional) : Strictly positive number (Default 1.2). Controls
%       cluster sizes in community_louvain function.
%       maxCommunityAttempts (optional) : Positive integer (Default 10).
%       Number of times to cluster with community_louvain. Best result is
%       taken.
%       phaseCounts (optional) : Positive integer (Default 125). Number of phase bins to include in minimal
%       model.
%       verbose (optional) : Logical (Default 0). Toggles drawing of intermediate steps    
%           
% OUTPUT : 
%       manifoldSpaceMap : t by 2 matrix of manifold space positions (community ID, phase) of each
%       observed data point.
%       manifoldSpaceTransitions : m by m matrix of manifold space
%       transition probabilities. m is the number of unique states.
%       manifoldSpaceWeights : m x t x c matrix of loadings of each
%       observed datapoint onto each manifold space phase bin. c is the
%       number of communities
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
if ~exist('stateDensity') || isempty(stateDensity) %Minimum occupancy of diffusion map for spectral analysis
    STATE_DENSITY = 1/4; 
else
    STATE_DENSITY = stateDensity;
end
if ~exist('minStateTime') || isempty(minStateTime) %Minimum length of a trajectory cluster
    MIN_STATE_TIME = 5;
else
    MIN_STATE_TIME = minStateTime;
end
if ~exist('gamma') || isempty(gamma) %Gamma value for maximum modularity clustering routine
    GAMMA = 1.2;
else
    GAMMA = gamma;
end
if ~exist('maxCommunityAttempts') || isempty(maxCommunityAttempts) %Number of times to redo clustering (best is choosen)
    MAX_COMMUNITY_ATTEMPTS = 10;
else
    MAX_COMMUNITY_ATTEMPTS = maxCommunityAttempts;
end
if ~exist('phaseCounts') || isempty(phaseCounts) %Number of phase bin to include in minimal model
    PHASE_COUNTS = 125;
else
    PHASE_COUNTS = phaseCounts;
end

%% Build over populated matrix for community finding
minT = 1;
probabilityMatrix = transitionMatrix;
denseProbabilityMatrix = probabilityMatrix;
while mean(sum(denseProbabilityMatrix ~= 0, 2)) <= size(probabilityMatrix,1) * STATE_DENSITY
    denseProbabilityMatrix = denseProbabilityMatrix * probabilityMatrix;
    
    minT = minT + 1;
end
denseProbabilityMatrix = denseProbabilityMatrix(1:end-minT,1:end-minT);
denseProbabilityMatrix = denseProbabilityMatrix ./ repmat(sum(denseProbabilityMatrix,2), 1, size(denseProbabilityMatrix,2));


connectivity = full(denseProbabilityMatrix + denseProbabilityMatrix');%abs(denseProbabilityMatrix) + abs(denseProbabilityMatrix)';
connectivity = connectivity ./ repmat(sum(connectivity,2), 1, size(connectivity,2));

if DISPLAY
    figure(4);
    clf;
    imagesc(sqrt(denseProbabilityMatrix));
    xlabel('State ID');
    xlabel('State ID');
    title('Asymmetric diffusion map');
    colormap(parula(256));
    c = colorbar;
    c.Label.String = ('Sqrt of Transition probability');
end
%%

[eigenModes, eigenvalues] = eigs(denseProbabilityMatrix, 50);
eigenvalues = (diag(eigenvalues));
eigenModes = (eigenModes);

ANGLE_EIGENMODE = find(imag(eigenvalues) > 0, 1);
allAngles = angle(eigenModes(:,ANGLE_EIGENMODE))';

[pcaBasis ~] = pca(embeddedTrace', 'NumComponents', 3);

if DISPLAY
    figure(5);
    clf;
    hold on;
    h = plot3(embeddedTrace'*pcaBasis(:,1), embeddedTrace'*pcaBasis(:,2), embeddedTrace'*pcaBasis(:,3));
    h.Color(4) = 0.1;
    scatter3(embeddedTrace(:,1:length(allAngles))'*pcaBasis(:,1), embeddedTrace(:,1:length(allAngles))'*pcaBasis(:,2), embeddedTrace(:,1:length(allAngles))'*pcaBasis(:,3), 32, allAngles);
    colormap(hsv(256));
    c = colorbar;
    c.Label.String = 'Phase (radians)';
    xlabel('DPCA 1');
    ylabel('DPCA 2');
    zlabel('DPCA 3');
    title('Phase of dominate complex eigenmode');
end

%%

connectionDistance = [];
if DISPLAY
    waitHandle = parfor_progressbar(size(connectivity,1), 'Calculating community distance');
end
for i = 1:size(connectivity,1)
    shiftedRows = [];
    for j = 0:100
        shiftedRows(j+1,:) = circshift(connectivity(i,:), [0, -j]);

        if j > 0
            shiftedRows(j+1,end - j + 1:end) = 0;
        end

        shiftedRows(j+1,:) = shiftedRows(j+1,:) / norm(shiftedRows(j+1,:));
    end

    minIndex = max(i-10,1);
    innerProducts = shiftedRows * connectivity';
    
    connectionDistance(i,:) = max(innerProducts);
    
    if DISPLAY
        waitHandle.iterate(1);
    end
end
if DISPLAY
    close(waitHandle);
end

connectivity = connectionDistance + connectionDistance';



pause(0.05);

%%
tic;
maxRawCommunity = [];
maxCost = 0;
if DISPLAY
    waitHandle = parfor_progressbar(MAX_COMMUNITY_ATTEMPTS, 'Computing communities');
end
for k = 1:MAX_COMMUNITY_ATTEMPTS
    rawCommunities = community_louvain(full(connectivity), GAMMA);

    ids = unique(rawCommunities);

    cost = 0;
    innerWeight = 0;
    outerWeight = 0;
    for i = 1:length(ids)
        selfIDs = rawCommunities == i;

        selfWeight = sum(denseProbabilityMatrix(selfIDs, selfIDs),2);

        innerWeight = innerWeight + sum(selfWeight.^2);
        outerWeight = outerWeight + sum((1 - selfWeight).^2);
    end
    cost = innerWeight / outerWeight;
    
    if cost > maxCost
        maxCost = cost;
        maxRawCommunity = rawCommunities;
    end
    
    if DISPLAY
        waitHandle.iterate(1);
    end
end
if DISPLAY
    close(waitHandle);
end

communities = rawCommunities;

neuronCounts = [];
for i = 1:max(communities)
    neuronCounts(i) = numel(find(communities == i));
end
[sortedCounts, countIndicies] = sort(neuronCounts);

for i = 1:max(communities)
    communities(communities == countIndicies(i)) = i + size(connectivity,1);
end
communities = communities - size(connectivity,1);

[communityIDs, indicies] = sort(communities);

sortedConnectivity = denseProbabilityMatrix(indicies,indicies);

numCategories = length(unique(communityIDs));


fullProbabilityMatrix = probabilityMatrix;
while mean(sum(fullProbabilityMatrix ~= 0, 2)) <= size(probabilityMatrix,1) * STATE_DENSITY
    fullProbabilityMatrix = fullProbabilityMatrix * probabilityMatrix;
end
%Resize using the larger minT from above, so that indices align
fullProbabilityMatrix = fullProbabilityMatrix(1:end-minT,1:end-minT);
fullProbabilityMatrix = fullProbabilityMatrix ./ repmat(sum(fullProbabilityMatrix,2), 1, size(fullProbabilityMatrix,2));



edges = find(diff(communities) ~= 0);
blockLengths = diff(edges);
[~, blockID] = min(blockLengths);
while blockLengths(blockID) < MIN_STATE_TIME    
    leftEdge = edges(blockID)+1;
    rightEdge = edges(blockID + 1);
    
    newID = communities(leftEdge-1);
    communities(leftEdge:rightEdge) = newID * ones(1,length(leftEdge:rightEdge));
    
    edges = find(diff(communities) ~= 0);
    blockLengths = diff(edges);
    [~, blockID] = min(blockLengths);
end

communityList = unique(communities);
numCategories = length(unique(communities));
for i = 1:numCategories
    communities(communities == communityList(i)) = i;
end

weights = [];
for i = 1:size(connectivity,2)
    weight = [];
    for j = 1:numCategories
        weight(j) = sum(fullProbabilityMatrix(i,find(communities == j)));
    end
    
    weights(i,:) = weight;
end
weights = weights ./ repmat(sum(weights,2),1,size(weights,2));

lineMap = lines(8);

%%
edges = linspace(-pi, pi, PHASE_COUNTS+1);

ANGLE_EPLISON = 0.01;
ANGLE_EIGENMODE = find(imag(eigenvalues) > 0, 1);
ANGLE_SIGMA = 2*pi/20;

barPositions = edges(1:end-1) + (edges(2) - edges(1))/2;

leftEdges = [];
rightEdges = [];
cutoffLeftEdges = [];
cutoffRightEdges = [];

angles = angle(eigenModes(:,ANGLE_EIGENMODE))';
totalBinCounts = histcounts(angles, edges);
communityBinCounts = [];
for i = 1:numCategories
    binCounts = histcounts(angles(communities == i), edges);
    communityBinCounts(:,i) = binCounts;
    binRatios = binCounts ./ totalBinCounts;
    
    sigma = 2; 
    xAxis = floor(-3*sigma):floor(3*sigma);
    filterSize = numel(xAxis);
    gaussian = exp(-xAxis.^2/(2*sigma^2))/(2*sigma^2*pi)^0.5;

    filteredTrace = binRatios;
    filteredTrace = cat(2, filteredTrace(end-filterSize+1:end), filteredTrace, filteredTrace(1:filterSize));
    filteredTrace = convn(filteredTrace, gaussian, 'same');
    binRatios = filteredTrace(:,filterSize+1:end-filterSize);
    
    [maxValue, maxIndex] = max(binRatios);
    zeroIndices = find(binRatios < maxValue / 4);
    leftValue = max(zeroIndices(zeroIndices < maxIndex));
    rightValue = min(zeroIndices(zeroIndices > maxIndex));
    if isempty(leftValue)
        leftValue = max(zeroIndices);
    end
    if isempty(rightValue)
        rightValue = min(zeroIndices);
    end
    if isempty(zeroIndices)
        leftValue = floor(length(barPositions)/2);
        rightValue = floor(length(barPositions)/2);
    end
    leftValue = barPositions(leftValue);
    rightValue = barPositions(rightValue);
    
    leftEdges(i) = wrapToPi(leftValue - ANGLE_SIGMA*2);
    rightEdges(i) = wrapToPi(rightValue + ANGLE_SIGMA*2);
    
    [maxValue, maxIndex] = max(binRatios);
    zeroIndices = find(binRatios < ANGLE_EPLISON);
    leftValue = max(zeroIndices(zeroIndices < maxIndex));
    rightValue = min(zeroIndices(zeroIndices > maxIndex));
    if isempty(leftValue)
        leftValue = max(zeroIndices);
    end
    if isempty(rightValue)
        rightValue = min(zeroIndices);
    end
    if isempty(zeroIndices)
        leftValue = floor(length(barPositions)/2);
        rightValue = floor(length(barPositions)/2);
    end
    
    cutoffLeftEdges(i) = leftValue;
    cutoffRightEdges(i) = rightValue;
end

%%
MAX_TAIL_LENGTH = 25;

allAngles = angle(eigenModes(:,ANGLE_EIGENMODE))';

trajectoryWeights = [];
trajectoryAngles = [];
for k = 1:numCategories  
    categories = k;
    category = find(ismember(communities, categories));
    sortedIndices = sort(category);
    
    startIndicesCommunity = find(diff([-1;sortedIndices]) > 1);
    endIndicesCommunity = [startIndicesCommunity(2:end)-1; length(sortedIndices)];
    
    finalIndices = [];
    failedTraces = [];
    for i = 1:length(startIndicesCommunity)
        minValue = max(sortedIndices(startIndicesCommunity(i)) - MAX_TAIL_LENGTH, 1);
        maxValue = min(sortedIndices(endIndicesCommunity(i)) + MAX_TAIL_LENGTH, length(communities));
        
        leftIndex = sortedIndices(startIndicesCommunity(i)) - minValue;
        rightIndex = (sortedIndices(endIndicesCommunity(i)) - sortedIndices(startIndicesCommunity(i))) + leftIndex;
        
        putativeIndicies = minValue:maxValue;
        
        if leftEdges(k) < rightEdges(k)
            midValues = find(allAngles(putativeIndicies) > leftEdges(k) & allAngles(putativeIndicies) < rightEdges(k) ...
                        & putativeIndicies > leftIndex + minValue & putativeIndicies < rightIndex + minValue);
            
            if isempty(midValues)
                failedTraces = [failedTraces, i];
                continue;
            end
            
            phaseOverlap = find(allAngles(putativeIndicies) > leftEdges(k) & allAngles(putativeIndicies) < rightEdges(k));
        
            phaseDiff = diff(phaseOverlap);
            
            minIndex = find(phaseOverlap == min(midValues));
            
            putativeIndex = [];
            if (minIndex > 1)
                putativeIndex = find(phaseDiff(1:minIndex-1) ~= 1, 1, 'last');
            end
            
            if ~isempty(putativeIndex)
                minValue = minValue + phaseOverlap(putativeIndex) + 1;
            end
            
            
            maxIndex = find(phaseOverlap == max(midValues));
            
            putativeIndex = [];
            if (maxIndex < length(phaseDiff))
                putativeIndex = find(phaseDiff(maxIndex:end) ~= 1, 1, 'first');
            end
            
            if ~isempty(putativeIndex)
                maxValue = maxValue - (length(putativeIndicies) - phaseOverlap(maxIndex + putativeIndex)) - 1;
            end
        else
            midValues = find((allAngles(putativeIndicies) > leftEdges(k) | allAngles(putativeIndicies) < rightEdges(k)) ...
                         & putativeIndicies > leftIndex + minValue & putativeIndicies < rightIndex + minValue);
                     
            if isempty(midValues)
                failedTraces = [failedTraces, i];
                continue;
            end
            
            phaseOverlap = find(allAngles(putativeIndicies) > leftEdges(k) | allAngles(putativeIndicies) < rightEdges(k));
        
            phaseDiff = diff(phaseOverlap);
            
            minIndex = find(phaseOverlap == min(midValues));
            
            putativeIndex = [];
            if (minIndex > 1)
                putativeIndex = find(phaseDiff(1:minIndex-1) ~= 1, 1, 'last');
            end
            
            if ~isempty(putativeIndex)
                minValue = minValue + phaseOverlap(putativeIndex) + 1;
            end
            
            
            maxIndex = find(phaseOverlap == max(midValues));
            
            putativeIndex = [];
            if (maxIndex < length(phaseDiff))
                putativeIndex = find(phaseDiff(maxIndex:end) ~= 1, 1, 'first');
            end
            
            if ~isempty(putativeIndex)
                maxValue = maxValue - (length(putativeIndicies) - phaseOverlap(maxIndex + putativeIndex)) - 1;
            end
        end
        
        leftIndex = sortedIndices(startIndicesCommunity(i)) - minValue;
        rightIndex = (sortedIndices(endIndicesCommunity(i)) - sortedIndices(startIndicesCommunity(i))) + leftIndex;
        
        putativeIndicies = minValue:maxValue;
        
        finalIndices = [finalIndices, minValue:maxValue];
        
        startIndicesCommunity(i) = minValue;
        endIndicesCommunity(i) = maxValue;
    end
    
    if isempty(finalIndices)
        continue;
    end
    
    startIndicesCommunity(failedTraces) = [];
    endIndicesCommunity(failedTraces) = [];
    
    sortedIndices = sort(unique(finalIndices))';
    
    categoryMatrix = probabilityMatrix(sortedIndices, sortedIndices);
    categoryMatrix(isnan(categoryMatrix)) = 0;
    for i = 1:size(categoryMatrix,1)-1
        if categoryMatrix(i,i+1) == 0
            if max(categoryMatrix(i,:)) > 0
                categoryMatrix(i,i+1) = max(categoryMatrix(i,:));
            else
                categoryMatrix(i,i+1) = 1;
            end
        end
    end
    categoryMatrix = categoryMatrix ./ repmat(sum(categoryMatrix,2), 1, size(categoryMatrix,2));
    categoryMatrix(isnan(categoryMatrix)) = 0;

    categoryWeights = weights(sortedIndices, :)';
    
    startIndices = [];
    endIndices = [];
    for i = 1:length(startIndicesCommunity)
        startIndices(i) = find(sortedIndices == startIndicesCommunity(i));
        endIndices(i) = find(sortedIndices == endIndicesCommunity(i));
    end
        
    angles = angle(eigenModes(sortedIndices,ANGLE_EIGENMODE))';% - angle(sum(inverseEigenVectors(firstImaginaryValue,:),1));
    startPhase = meanangle(angles(startIndices) * 180 / pi) * pi / 180;

    smoothedAngles = [];
    times = [];
    for i = 1:length(startIndices)
        startIndex = startIndices(i);
        if i == length(startIndices)
            endIndex = length(sortedIndices);
        else
            endIndex = max(startIndices(i+1)-1, endIndices(i));
        end
        
        sigma = 5; 
        xAxis = floor(-3*sigma):floor(3*sigma);
        filterSize = numel(xAxis);
        gaussian = exp(-xAxis.^2/(2*sigma^2))/(2*sigma^2*pi)^0.5;

        filteredTrace = unwrap(angles(startIndex:endIndex));
        filteredTrace = cat(2, repmat(filteredTrace(:,1)*1,[1,filterSize]), filteredTrace, repmat(filteredTrace(:,end)*1,[1,filterSize]));
        filteredTrace = convn(filteredTrace, gaussian, 'same');
        smoothedAngles(startIndex:endIndex) = filteredTrace(:,filterSize+1:end-filterSize);
    end
    
    diffAngles = diff(smoothedAngles);
    negativeIndicies = find(diffAngles < 0) + 1;
    
    angles(negativeIndicies) = NaN;
    
    % Is full loop
    if leftEdges(k) == rightEdges(k)
        angles(communities(sortedIndices) ~= k) = nan;
    end
    angles(communities(sortedIndices) ~= k) = nan;
    
    phaseVelocities = diff(angles);
    phaseVelocities(:, end+1) = phaseVelocities(:, end);
    

    sigma = ANGLE_SIGMA;
    for i = 1:length(barPositions)
        theta = barPositions(i);        
        trajectoryAngles(i, k) = theta;
        
        if 1%(theta >= startAngle && theta <= endAngle) || ((theta <= endAngle || theta >= startAngle) && startAngle >= endAngle)
            x = angdiff(angles, repmat(theta, 1, length(angles)));

            binMids = edges(1:end-1) + (edges(2) - edges(1))/2;
            density = interp1(binMids, communityBinCounts(:,k), wrapToPi(theta));

            averageWeights = exp(-x.^2/(2*sigma^2))/(2*sigma^2*pi)^0.5;
            
            averageWeights(isnan(averageWeights)) = 0;
            averageWeights = averageWeights ./ nansum(averageWeights);
            
            if sum(isnan(averageWeights)) > 0
                averageWeights = zeros(size(averageWeights));
            end

            allWeights = zeros(1,size(probabilityMatrix,1));
            allWeights(sortedIndices) = averageWeights;
            trajectoryWeights(:,i,k) = allWeights;
        else
            allWeights = nan(1,size(probabilityMatrix,1));
            trajectoryWeights(:,i,k) = allWeights;   
        end
    end
end

manifoldSpaceWeights = trajectoryWeights;


%% Build manifold space

angles = allAngles;

communityThetas = edges(1:end-1) + (edges(2) - edges(1))/2;
stateMap = [communities, ceil((angles - edges(1)) / (2*pi) * (length(edges) - 1))'];
[uniqueStates, stateUniqueMap, stateIDs] = unique(stateMap, 'rows');
stateBackMap = stateMap(stateUniqueMap,:);

% steadyState = [];
% steadyStateSpace = [];
% for i = 1:size(uniqueStates,1)
%     steadyState(i) = sum(stateIDs == i);
%     
%     steadyStateSpace(stateBackMap(i,2), stateBackMap(i,1)) = sum(stateIDs == i);
% end
% steadyStateSpace = steadyStateSpace ./sum(steadyStateSpace(:));

transitionProbabilities = zeros(length(uniqueStates));
for i = 2:size(stateMap,1)
    transitionProbabilities(stateIDs(i - 1), stateIDs(i)) = transitionProbabilities(stateIDs(i - 1), stateIDs(i)) + 1;
end

transitionProbabilities = transitionProbabilities ./repmat(sum(transitionProbabilities,2),1,size(transitionProbabilities,2));

for i = 1:size(uniqueStates,1)
    if transitionProbabilities(i,i) == 1
        transitionProbabilities(i,i+1) = 1;
        transitionProbabilities(i,i) = 0;
    end
end

manifoldSpaceTransitions = transitionProbabilities;
manifoldSpaceMap = stateMap;

%% Plot communities

if DISPLAY
    [pcaBasis, ~] = pca(embeddedTrace', 'NumComponents', 3);

    figure(6);
    clf;
    h = plot3(embeddedTrace'*pcaBasis(:,1), embeddedTrace'*pcaBasis(:,2), embeddedTrace'*pcaBasis(:,3), 'k');
    h.Color(4) = 0.1;
    hold on;
    linesMaps = lines(size(manifoldSpaceWeights,3));
    for i = 1:size(manifoldSpaceWeights,3)
        startPoint = cutoffLeftEdges(i);
        endPoint = cutoffRightEdges(i);

        if startPoint < endPoint
            plotPhases = startPoint:endPoint;
        else
            plotPhases = [startPoint:size(manifoldSpaceWeights,2) 1:endPoint];
        end

        thisTrajectory = embeddedTrace(:,1:end-1) * manifoldSpaceWeights(:,plotPhases,i);
        thisPoints = find(manifoldSpaceMap(:,1) == i);

        h = scatter3(embeddedTrace(:,thisPoints)'*pcaBasis(:,1), embeddedTrace(:,thisPoints)'*pcaBasis(:,2), embeddedTrace(:,thisPoints)'*pcaBasis(:,3), 32, linesMaps(i,:));
        h.MarkerEdgeAlpha = 0.1;
        plot3(thisTrajectory'*pcaBasis(:,1),thisTrajectory'*pcaBasis(:,2),thisTrajectory'*pcaBasis(:,3), 'LineWidth', 3, 'Color', linesMaps(i,:));
    end
    xlabel('DPCA 1');
    ylabel('DPCA 2');
    zlabel('DPCA 3');
    title('Loop identification (Colors)');
end
