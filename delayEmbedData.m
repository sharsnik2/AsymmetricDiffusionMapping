function [delayedTrace, embeddedTime] = delayEmbedData(activityTrace, delayTime, delayCount, useDerivative, useZScore, smoothSigma, shouldDetrend, verbose)
% findPhaseSpace : This function pre-processes a multivariate time series
% datastream and applies delay embedding in an effort to reconstruct phase
% space.
%
% INPUT : 
%       activityTrace : n by t time series data where n are the number of
%           dimensions and t are the temporally sequenced observations
%       delayTime : Integer. Number of times to delay embed
%       delayCount : Integer. Number of timesteps to delay per embedding
%       useDerivative (optional) : Logical (Default 0). Toggles appending of the derivatives of the time series data to the delay embedded space
%       useZScore (optional) : Logical (Default 0). Toggles zscoring of the delayed data (both activity and derivatives)
%       shouldDetrend (optional) : Logical (Default 0). Toggles detrending of the data prior to delay embedding
%       smoothSigma (optional) : Positive double (Default 0). Sigma of Gaussian (in timesteps) of smoothing kernel to be applied to the data
%       verbose (optional) : Logical (Default 0). Toggles drawing of intermediate steps
%           
% OUTPUT : 
%       delayedTrace : Delay embedding of activity trace. Size of this trace is:
%           (n * delayCount * [2 if useDerivative is true]) by
%           (t - delayCount * delayTime - [1 if useDerivative is true]).
%           Rows of this matrix are order from top to bottom as:
%           activity(1:n,t), derivative(1,:n,t) if useDerivative is true,
%           activity(1:n,t-1*delayTime), derivative(1,:n,t-1*delayTime) if useDerivative is true,
%           etc until:
%           activity(1:n,t-delayCount*delayTime), derivative(1,:n,t-delayCount*delayTime) if useDerivative is true,
%       embeddedTime: The total time offset induced by the embedding. delayTime * delayCount + useDerivative
%           This can be used to convert from the original timesteps, t, to delay
%           embedded timesteps, tao, by: t = tao - embeddedTime
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
if ~exist('useDerivative') || isempty(useDerivative) %Should the phase space be appended with the derivatives?
    USE_DERIVATIVES = 1;
else
    USE_DERIVATIVES = useDerivative;
end
if ~exist('useZScore') || isempty(useZScore) %Should the data be zscored before delay embedding?
    USE_ZSCORE = 0;
else
    USE_ZSCORE = useZScore;
end
if ~exist('shouldDetrend') || isempty(shouldDetrend) %Should the data be detrended?
    DETREND = 0;
else
    DETREND = shouldDetrend;
end
if ~exist('smoothSigma') || isempty(smoothSigma)%Amount of smoothing to apply to the time series
    SMOOTH_SIGMA = 0;
else
    SMOOTH_SIGMA = smoothSigma;
end

%Delay embedding options
DELTA_T = delayTime;
NUM_T = delayCount;


%% Start method
neuronData = activityTrace;

if DETREND
    neuronDataDetrended = detrend(neuronData);
else
    neuronDataDetrended = neuronData;
end
if SMOOTH_SIGMA > 0
    sigma = SMOOTH_SIGMA; 
    x = floor(-3*sigma):floor(3*sigma);
    filterSize = numel(x);
    gaussian = exp(-x.^2/(2*sigma^2))/(2*sigma^2*pi)^0.5;

    filteredTrace = neuronDataDetrended;
    filteredTrace = cat(2, repmat(filteredTrace(:,1),[1,filterSize]), filteredTrace, repmat(filteredTrace(:,end),[1,filterSize]));
    filteredTrace = convn(filteredTrace, gaussian, 'same');
    neuronDataDetrended = filteredTrace(:,filterSize+1:end-filterSize);
end

neuronDataDiff = [];
if USE_DERIVATIVES
    filteredTrace = diff(neuronDataDetrended,1,2);
    
    neuronDataDiff(:,:) = [zeros(size(neuronDataDetrended,1)), filteredTrace];
end

%%
embeddedTime = NUM_T*DELTA_T + USE_DERIVATIVES;
stateSpace = [];
index = 1;
for i = 1 + embeddedTime:size(neuronDataDetrended, 2)
    thisVector = [];
    
    for j = 0:NUM_T
        thisVector = [thisVector; neuronDataDetrended(:,i - j*DELTA_T)];
        
        if USE_DERIVATIVES
            thisVector = [thisVector; neuronDataDiff(:,i - j*DELTA_T)];
        end
    end
    
    stateSpace(:,index) = thisVector;
    index = index + 1;
end

if USE_ZSCORE
    stateSpace = zscore(stateSpace, 0, 2);
end

delayedTrace = stateSpace;

if DISPLAY
    figure(3);
    clf;
    [~, pcaScores] = pca(delayedTrace', 'NumComponents', 3);
    startPCA = 1;
    score = pcaScores(:,startPCA:startPCA+2);
    indices = 1:size(score,1);
%     vectors = [];
%     for i = 1:size(score, 1)-1
%         vectors(i,:) = score(i + 1,:) - score(i,:);
%     end
%     vectors(size(score, 1),:) = [0, 0, 0];
    hold on;
%     quiver3(score(indices,1),score(indices,2),score(indices,3),vectors(:,1),vectors(:,2),vectors(:,3),3);
    plot3(score(indices,1),score(indices,2),score(indices,3));
    colormap(jet(256));
    caxis([1 8])
    xlabel('DPCA 1');
    ylabel('DPCA 2');
    zlabel('DPCA 3');
    title('PCA of delay embedded data');
end


