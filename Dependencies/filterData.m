
function [filteredData] = filterData(data, sigma, kernel, noFlip, paddingMode)
% paddingMode = 0 -> Pad with repeated values
% paddingMode = 1 -> Pad with 0s
% paddingMode = 2 -> Pad circularly

if ~exist('noFlip', 'var') || isempty('noFlip')
    noFlip = false;
end

if ~exist('padZeros', 'var') || isempty('padZeros')
    paddingMode = 1;
end

flipped = 0;
if ~noFlip && size(data,2) < size(data,1)
    data = data';
    flipped = 1;
end

if ~exist('kernel', 'var') || isempty(kernel) || strcmpi(kernel, 'gaussian')
    x = floor(-3*sigma):floor(3*sigma);
    filterSize = numel(x);
    kernel = exp(-x.^2/(2*sigma^2))/(2*sigma^2*pi)^0.5;
elseif strcmpi(kernel, 'halfGaussian')
    x = floor(-3*sigma):floor(3*sigma);
    filterSize = numel(x);
    kernel = exp(-x.^2/(2*sigma^2))/(2*sigma^2*pi)^0.5;
    kernel(x <= 0) = 0;    
else
    filterSize = numel(kernel);
end

filteredTrace = data;
if paddingMode == 2    
    filteredTrace = cat(2, filteredTrace(:,end-filterSize+1:end), filteredTrace, filteredTrace(:,1:filterSize));
elseif paddingMode == 1
    filteredTrace = cat(2, repmat(filteredTrace(:,1)*0,[1,filterSize]), filteredTrace, repmat(filteredTrace(:,end)*0,[1,filterSize]));
else
    filteredTrace = cat(2, repmat(filteredTrace(:,1)*1,[1,filterSize]), filteredTrace, repmat(filteredTrace(:,end)*1,[1,filterSize]));
end
filteredTrace = convn(filteredTrace, kernel, 'same');
filteredData = filteredTrace(:,filterSize+1:end-filterSize);

if flipped == 1
    filteredData = filteredData';
end