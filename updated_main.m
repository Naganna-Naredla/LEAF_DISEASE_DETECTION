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
for i = 1:length(subfolders)
    imageFiles = dir(fullfile(subfolders(i).folder, subfolders(i).name, '*.jpg'));
    numImages = numImages + length(imageFiles);
end

% --- Determine LBP Feature Size Dynamically ---
% Read a sample image to determine LBP feature size
sampleImagePath = fullfile(subfolders(1).folder, subfolders(1).name, '*.jpg');
sampleImageFile = dir(sampleImagePath);
sampleImg = imread(fullfile(sampleImageFile(1).folder, sampleImageFile(1).name));
sampleGrayImg = rgb2gray(sampleImg);
sampleLBPFeatures = extractLBPFeatures(sampleGrayImg, 'Upright', false, 'Normalization', 'L2');
lbpFeatureSize = length(sampleLBPFeatures);

% Preallocate for histogram features (256 bins for each channel) and LBP
histogramFeatures = zeros(numImages, 256 * 3); % RGB histograms
lbpFeatures = zeros(numImages, lbpFeatureSize); % LBP features
labels = categorical(); % Initialize labels

% Loop through subfolders and extract features
imageIndex = 1;
for i = 1:length(subfolders)
    imageFiles = dir(fullfile(subfolders(i).folder, subfolders(i).name, '*.jpg'));
    for j = 1:length(imageFiles)
        % Read the image
        img = imread(fullfile(imageFiles(j).folder, imageFiles(j).name));
        
        % --- Histogram Features ---
        redHist = imhist(img(:, :, 1), 256) / numel(img(:, :, 1));
        greenHist = imhist(img(:, :, 2), 256) / numel(img(:, :, 2));
        blueHist = imhist(img(:, :, 3), 256) / numel(img(:, :, 3));
        histogramFeatures(imageIndex, :) = [redHist', greenHist', blueHist'];
        
        % --- LBP Features ---
        grayImg = rgb2gray(img);
        lbpFeatures(imageIndex, :) = extractLBPFeatures(grayImg, 'Upright', false, 'Normalization', 'L2');
        
        % Label
        sanitizedLabel = regexprep(subfolders(i).name, '[^\w'']', '');
        labels = [labels; categorical({sanitizedLabel})];
        
        % Increment image index
        imageIndex = imageIndex + 1;
    end
end

% --- Histogram Classifier ---
fprintf('\n--- Results for Color Histograms ---\n');
runClassification(histogramFeatures, labels, 'Color Histograms');

% --- LBP Classifier ---
fprintf('\n--- Results for LBP ---\n');
runClassification(lbpFeatures, labels, 'LBP');

% --- Helper Function to Run Classifiers ---
function runClassification(features, labels, featureType)
    % Split data into training and testing sets (80% train, 20% test)
    cv = cvpartition(length(labels), 'HoldOut', 0.2);
    Xtrain = features(cv.training, :);
    Ytrain = labels(cv.training);
    Xtest = features(cv.test, :);
    Ytest = labels(cv.test);

    % Remove constant columns (zero variance)
    constantCols = var(Xtrain) == 0;
    Xtrain(:, constantCols) = [];
    Xtest(:, constantCols) = [];

    % --- Classifiers ---
    classifiers = {
        'KNN', @fitcknn;
        'Decision Tree', @fitctree;
        'Random Forest', @fitcensemble;
        'Logistic Regression', @(X, Y) fitcecoc(X, Y, 'Learners', 'linear', 'Coding', 'onevsall');
        'Naive Bayes', @fitcnb;
    };

    for c = 1:size(classifiers, 1)
        modelName = classifiers{c, 1};
        modelFunc = classifiers{c, 2};
        
        % Try to train and predict; catch errors
        try
            model = modelFunc(Xtrain, Ytrain);
            predictions = predict(model, Xtest);
            evaluateModel(predictions, Ytest, [featureType ' - ' modelName]);
        catch ME
            fprintf('Error with %s: %s\n', modelName, ME.message);
        end
    end
end

% --- Helper Function to Evaluate Models ---
function evaluateModel(Ypred, Ytest, modelName)
    % Convert inputs to categorical if they are not already
    if ~iscategorical(Ypred)
        Ypred = categorical(Ypred);
    end
    if ~iscategorical(Ytest)
        Ytest = categorical(Ytest);
    end

    % Calculate accuracy
    accuracy = sum(Ypred == Ytest) / numel(Ytest);

    % Confusion Matrix
    cm = confusionmat(Ytest, Ypred);

    % Precision, Recall, and F1-Score
    precision = diag(cm) ./ sum(cm, 1)'; % Precision per class
    recall = diag(cm) ./ sum(cm, 2);     % Recall per class
    f1score = 2 * (precision .* recall) ./ (precision + recall);

    % Handle NaN values in precision, recall, and F1 (e.g., if no true positives)
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    f1score(isnan(f1score)) = 0;

    % Average Precision, Recall, and F1-Score
    avgPrecision = mean(precision);
    avgRecall = mean(recall);
    avgF1Score = mean(f1score);

    % Print Results
    fprintf('%s:\n', modelName);
    fprintf('  Accuracy: %.4f\n', accuracy);
    fprintf('  Precision: %.4f\n', avgPrecision);
    fprintf('  Recall: %.4f\n', avgRecall);
    fprintf('  F1-Score: %.4f\n\n', avgF1Score);

    % Plot Confusion Matrix
    figure;
    confusionchart(Ytest, Ypred, 'Title', [modelName ' Confusion Matrix']);
end
