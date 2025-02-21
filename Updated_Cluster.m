clc;
clear;
close all;

% Define the folder containing images
imageFolder = '.\PlantVillage';

% Get all subfolders (categories)
subfolders = dir(imageFolder);
subfolders = subfolders([subfolders.isdir]);
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Exclude '.' and '..'

% Initialize arrays for features and labels
numImages = 0;
folderNames = {subfolders.name}; % Store folder names
imageCounts = zeros(1, length(subfolders)); % Image counts for plotting

% Count the total number of images
for i = 1:length(subfolders)
    imageFiles = dir(fullfile(subfolders(i).folder, subfolders(i).name, '*.jpg'));
    numImages = numImages + length(imageFiles);
    imageCounts(i) = length(imageFiles); % Store the image count for the folder
end

% Preallocate memory for features and labels
features = zeros(numImages, 256*3); % RGB histograms (256 bins each for R, G, B)
labels = categorical();

% Extract features from images
imageIndex = 1;
for i = 1:length(subfolders)
    imageFiles = dir(fullfile(subfolders(i).folder, subfolders(i).name, '*.jpg'));
    for j = 1:length(imageFiles)
        % Read the image
        img = imread(fullfile(imageFiles(j).folder, imageFiles(j).name));
        
        % Calculate histograms for R, G, B channels
        redHist = imhist(img(:, :, 1), 256);
        greenHist = imhist(img(:, :, 2), 256);
        blueHist = imhist(img(:, :, 3), 256);
        
        % Normalize histograms
        redHist = redHist / sum(redHist);
        greenHist = greenHist / sum(greenHist);
        blueHist = blueHist / sum(blueHist);
        
        % Concatenate histograms to form the feature vector
        features(imageIndex, :) = [redHist', greenHist', blueHist'];
        
        % Assign label
        sanitizedLabel = regexprep(subfolders(i).name, '[^\w'']', '');
        labels = [labels; categorical({sanitizedLabel})];
        
        % Increment the image index
        imageIndex = imageIndex + 1;
    end
end

% Apply PCA for dimensionality reduction
[coeff, pcaFeatures] = pca(features);
reducedFeatures = pcaFeatures(:, 1:50); % Top 50 PCA components

% Convert categorical labels to numeric for clustering evaluation
trueLabels = grp2idx(labels);

% --- K-Means Clustering ---
numClusters = numel(unique(trueLabels)); % Set number of clusters as the number of classes
[idxKMeans, ~] = kmeans(reducedFeatures, numClusters, 'Distance', 'sqeuclidean', 'Replicates', 10);

% Evaluate K-Means
ARI_kmeans = adjustedRandIndex(trueLabels, idxKMeans); % Adjusted Rand Index
fprintf('K-Means Adjusted Rand Index: %.4f\n', ARI_kmeans);

% Visualize K-Means clusters in 2D (using first two PCA components)
figure;
scatter(reducedFeatures(:, 1), reducedFeatures(:, 2), 10, idxKMeans, 'filled');
title('K-Means Clustering');
xlabel('PCA Component 1');
ylabel('PCA Component 2');

% --- DBSCAN Clustering ---
epsilon = 0.5; % Neighborhood radius
minPts = 10;  % Minimum number of points to form a cluster
idxDBSCAN = dbscan(reducedFeatures, epsilon, minPts);

% Evaluate DBSCAN
ARI_dbscan = adjustedRandIndex(trueLabels, idxDBSCAN);
fprintf('DBSCAN Adjusted Rand Index: %.4f\n', ARI_dbscan);

% Visualize DBSCAN clusters in 2D
figure;
scatter(reducedFeatures(:, 1), reducedFeatures(:, 2), 10, idxDBSCAN, 'filled');
title('DBSCAN Clustering');
xlabel('PCA Component 1');
ylabel('PCA Component 2');

% --- Helper Function: Adjusted Rand Index ---
function ARI = adjustedRandIndex(trueLabels, predLabels)
    % Calculate confusion matrix
    contingencyMatrix = confusionmat(trueLabels, predLabels);
    
    % Sum over rows and columns
    sumRows = sum(contingencyMatrix, 2);
    sumCols = sum(contingencyMatrix, 1);
    
    % Total number of samples
    total = sum(contingencyMatrix(:));
    
    % Combinatorial calculations
    sumComb = sum(nchoosekVectorized(contingencyMatrix(:), 2));
    rowComb = sum(nchoosekVectorized(sumRows, 2));
    colComb = sum(nchoosekVectorized(sumCols, 2));
    
    % Expected index and maximum index
    expectedIndex = (rowComb * colComb) / nchoosek(total, 2);
    maxIndex = 0.5 * (rowComb + colComb);
    
    % Adjusted Rand Index
    ARI = (sumComb - expectedIndex) / (maxIndex - expectedIndex);
end

% Helper Function: Vectorized nchoosek
function result = nchoosekVectorized(array, k)
    % Vectorized calculation of nchoosek for an array
    result = array .* (array - 1) / 2;
    result(array < k) = 0; % Set invalid values to 0
end
