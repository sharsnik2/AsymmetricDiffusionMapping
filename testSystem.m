% testSystem : This script is a sanity check for the packaged functions
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


%% Parameters below have been fine tuned for this seed. May need to retune for best results on other seeds
rng(0)


%% Simulate the two neuron system illustarted in the appendix
% Simulate without noise to find limit cycles
t1 = 1;
t2 = 6;
w11 = 8;
w12 = -6;
w21 = 16;
w22 = -2;
b1 = .34;
b2 = 2.5;

limX = [0 1];
limY = [0.2 1];

xProbe = linspace(limX(1),limX(2),20);
yProbe = linspace(limY(1),limY(2),20);
[xGrid,yGrid] = meshgrid(xProbe, yProbe);


fx = @(x)(-x(:,1) + sigmf(w11*x(:,1) + w12*x(:,2) - b1, [1 0]))/t1;
fy = @(x)(-x(:,2) + sigmf(w21*x(:,1) + w22*x(:,2) - b2, [1 0]))/t2;

dt = 1/100;

limit1x = 0.1;
limit1y = 0.5;
for t = 2:100000
    limit1x(t) = limit1x(t-1) + ((-limit1x(t-1) + sigmf(w11*limit1x(t-1) + w12*limit1y(t-1) - b1, [1 0]))/t1)*dt;
    limit1y(t) = limit1y(t-1) + ((-limit1y(t-1) + sigmf(w21*limit1x(t-1) + w22*limit1y(t-1) - b2, [1 0]))/t2)*dt;
end

limit2x = 0.5;
limit2y = 0.1;
for t = 2:100000
    limit2x(t) = limit2x(t-1) + ((-limit2x(t-1) + sigmf(w11*limit2x(t-1) + w12*limit2y(t-1) - b1, [1 0]))/t1)*dt;
    limit2y(t) = limit2y(t-1) + ((-limit2y(t-1) + sigmf(w21*limit2x(t-1) + w22*limit2y(t-1) - b2, [1 0]))/t2)*dt;
end

%% Simulate network with noise

MAX_TIME = 1000000;
NOISE_SIGMA = 0.1;
TRACE_LENGTH = 3000;

x = zeros(1,MAX_TIME);
y = zeros(1,MAX_TIME);
x(1) = 0.5;
y(1) = 0.1;
for t = 2:MAX_TIME
    x(t) = x(t-1) + ((-x(t-1) + sigmf(w11*x(t-1) + w12*y(t-1) - b1, [1 0]))/t1 + normrnd(0, NOISE_SIGMA))*dt;
    y(t) = y(t-1) + ((-y(t-1) + sigmf(w21*x(t-1) + w22*y(t-1) - b2, [1 0]))/t2 + normrnd(0, NOISE_SIGMA))*dt;
end

%% Plot data as a time series

finalTraceIndices = MAX_TIME-TRACE_LENGTH*10+1:10:MAX_TIME;

figure(1);
clf;
subplot(2,1,1)
plot(zscore(x(finalTraceIndices)), 'r','LineWidth',1);
ylabel('Activity of neuron A (au)');
xlabel('Time (timesteps)');
subplot(2,1,2)
plot(zscore(y(finalTraceIndices)), 'b','LineWidth',1);
ylabel('Activity of neuron B (au)');
xlabel('Time (timesteps)');
suptitle('Activity trace of toy system');

%% Plot data as a a phase plot

traceColor = lines(2);

figure(2);
clf;
xlim(limX);
ylim(limY);
hold on;
quiver(xGrid(:),yGrid(:),fx([xGrid(:),yGrid(:)]),fy([xGrid(:),yGrid(:)]),'k','linewidth',0.5);
plot(x(finalTraceIndices), y(finalTraceIndices), 'Color', traceColor(2,:),'LineWidth',2);
plot(limit1x(end-10000:end),limit1y(end-10000:end),'color','k','linewidth',4);
plot(limit2x(end-10000:end),limit2y(end-10000:end),'color','k','linewidth',4);
xlabel('Activity of neuron A (au)');
ylabel('Activity of neuron B (au)');
title('Phase protrait of toy system');

%% Asymmetric Diffusion Map Method
% For our test we'll use the method on only a subset of the full time series
finalTrace = x(finalTraceIndices);

% Delay embed only the x part of the trace using 3 embeddings with a tao of
% 10 frames each. Plot PCA of the output.
[embeddedTrace, embeddedTime] = delayEmbedData(finalTrace, 5, 10, 1, 1, 0.5, 0, 1);


%% Build diffusion map
transitionMatrix = buildAsymmetricDiffusionMap(embeddedTrace, 5, 4, 10, 1);

%% Cluster diffusion map and use phase information to build a minimal model
[manifoldSpaceMap, manifoldSpaceTransitions, manifoldSpaceWeights] = buildMinimalModel(transitionMatrix, embeddedTrace, 1/4, 5, 1.2, 10, 100, 1);

%% Simulate the minimal model, and compare returns times
simulatedActivity = simulateMinimalModel(manifoldSpaceMap, manifoldSpaceTransitions, manifoldSpaceWeights, embeddedTrace, 1000000, 1);

plotIndicies = 5000+1:15000;
plotTrace = simulatedActivity(plotIndicies,:)';

realData = filterData(x(1,500:10:end), 20);
simData = filterData(simulatedActivity(:,1), 20);

figure(9);
clf;
subplot(2,1,1);
plot(realData(1:length(plotIndicies)));
ylabel('Activity of neuron A (au)');
xlabel('Time (timesteps)');
title('Observed');
subplot(2,1,2);
plot(simData(1:length(plotIndicies)));
ylabel('Activity of neuron A (au)');
xlabel('Time (timesteps)');
title('Simulation');
suptitle('Simulated activity trace of toy system');


[~,realPeaks] = findpeaks(-realData);
realDwellTimes = diff(realPeaks);

[~,simPeaks] = findpeaks(-simData);
simDwellTimes = diff(simPeaks);


figure(10);
clf;
hold on;

[realFrequency,realDistribution] = ksdensity(realDwellTimes);
[simFrequency,simDistribution] = ksdensity(simDwellTimes);

allDistribution = [realDistribution, simDistribution];

residuePoints = linspace(min(allDistribution), max(allDistribution), 100);

realDistribution = interp1q(realDistribution', realFrequency', residuePoints');
simDistribution = interp1q(simDistribution', simFrequency', residuePoints');

realDistribution = realDistribution / sqrt(nansum(realDistribution.^2));
simDistribution = simDistribution / sqrt(nansum(simDistribution.^2));

r2 = nansum(realDistribution .* simDistribution);

plot(realDistribution)
plot(simDistribution)
ylabel('Frequency');
xlabel('Return time of system (timesteps)');
title(['Return time distributions of simulation (r^2 = ' num2str(r2) ')']);
