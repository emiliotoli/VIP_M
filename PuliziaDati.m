%% organizzazione file

% Percorsi
csvFile = 'train_small.csv'; % File CSV con le etichette
imagesFolder = '../train_set'; % Cartella con tutte le immagini
outputFolder = '../output'; % Cartella di output organizzata
outlierFolder = fullfile(outputFolder, '251'); % Cartella per gli outlier
% 
% rmdir(outlierFolder, 's');
% rmdir(outputFolder, 's');


% Creazione delle cartelle di output
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
if ~exist(outlierFolder, 'dir')
    mkdir(outlierFolder);
end

% Leggi il CSV
data = readtable(".." + "/Annotazioni/" + csvFile, 'ReadVariableNames', false);
%Nomi immagini
imageNames = data.Var1;
%etichette
classLabels = data.Var2;

% Organizza immagini per classe
uniqueClasses = unique(classLabels);
for i = 1:length(uniqueClasses)
    classFolder = fullfile(outputFolder, num2str(uniqueClasses(i)));
    if ~exist(classFolder, 'dir')
        mkdir(classFolder);
    end
end

% Sposta le immagini nelle cartelle delle classi
for i = 1:height(data)
    imageName = imageNames{i};
    classLabel = classLabels(i);
    srcPath = fullfile(imagesFolder, imageName);
    dstPath = fullfile(outputFolder, num2str(classLabel), imageName);
    if exist(srcPath, 'file')
        copyfile(srcPath, dstPath);
    else
        fprintf('Immagine non trovata: %s\n', srcPath);
    end
end

fprintf('Immagini etichettate organizzate per classe.\n');

%% Gestione feature e modello

% Carica modello pre-addestrato per estrazione feature
net = resnet18; % Modello pre-addestrato
%layer = 'avg_pool';
layer = 'pool5'; % Strato per feature extraction (corretto per ResNet18)
%analyzeNetwork(net);


% Analisi di coerenza interna per classe: pulizia train set
for classIdx = 1:length(uniqueClasses)
    classFolder = fullfile(outputFolder, num2str(uniqueClasses(classIdx)));
    imageFiles = dir(fullfile(classFolder, '*.jpg'));
    
    features = [];
    filePaths = {};
    variances = [];
    for i = 1:length(imageFiles)
        imgPath = fullfile(classFolder, imageFiles(i).name);
        img = imread(imgPath);
        img = imresize(img, [224, 224]); % Resizing per il modello
        
        % Feature dal modello
        feature = activations(net, img, layer, 'OutputAs', 'rows');
        features = [features; feature]; %#ok<AGROW>
        filePaths{end+1} = imgPath; %#ok<AGROW>
        
        % Calcolo varianza dei pixel
        variances = [variances; var(double(img(:)))]; %#ok<AGROW>
    end

    [~, C] = kmeans(features, 1, 'Replicates', 5, 'Distance', 'sqeuclidean');
    distances = pdist2(features, C);
    mean_dst = mean(distances);
    std_dst = std(distances);
    alpha = 1.2;
    threshold = mean_dst + alpha*std_dst;

    % Calcolo del centroide della classe
    %centroid = mean(features, 1);
    %distances = vecnorm(features - centroid, 2, 2); % Distanza euclidea

    % Applica DBSCAN
    %epsilon = 0.5; % Distanza massima tra i punti per considerare lo stesso cluster
    %minPts = 1;    % Numero minimo di punti per formare un cluster
    %labels = dbscan(features, epsilon, minPts, 'Distance', 'euclidean');

    % Identifica gli outlier (label == -1)
    %outliers = (labels == -1);

    segmentOutliers = false(size(imageFiles));
    distanceThreshold = false(size(imageFiles));
    for i = 1:length(imageFiles)
        %% Segmentazione
        imgPath = fullfile(classFolder, imageFiles(i).name);
        img = imread(imgPath);
        % Converti in scala di grigi e binarizza
        grayImg = rgb2gray(img);
        binaryImg = imbinarize(grayImg, 'adaptive', 'Sensitivity', 0.5);
        % Controlla area segmentata
        segmentedArea = sum(binaryImg(:));
        totalArea = numel(binaryImg);
        % Se l'area segmentata è troppo piccola o troppo grande, è un outlier
        if segmentedArea < 0.1 * totalArea || segmentedArea > 0.98 * totalArea
            segmentOutliers(i) = true;
        end

        %% distanza centroide
        if(distances(i)>=threshold)
            distanceThreshold(i)=true;
        end

        %% Calcolo della media e varianza del colore sulle zone segmentate
        segmentedPixels = double(img) .* double(binaryImg); % Applica la maschera
    
        % Seleziona solo i pixel segmentati (non zero)
        segmentedPixelsR = segmentedPixels(:, :, 1);
        segmentedPixelsG = segmentedPixels(:, :, 2);
        segmentedPixelsB = segmentedPixels(:, :, 3);
    
        % Calcola la media e la varianza solo sui pixel segmentati
        meanColor = [
            mean(segmentedPixelsR(binaryImg > 0)), ...
            mean(segmentedPixelsG(binaryImg > 0)), ...
            mean(segmentedPixelsB(binaryImg > 0))
        ];
    
        varColor = [
            var(segmentedPixelsR(binaryImg > 0)), ...
            var(segmentedPixelsG(binaryImg > 0)), ...
            var(segmentedPixelsB(binaryImg > 0))
        ];

        % Salva i risultati
        meanColors(i, :) = meanColor(:)';
        varColors(i, :) = varColor(:)';

    end

    % Calcola la media e la varianza della classe
    classMeanColor = mean(meanColors, 1);
    classVarColor = mean(varColors, 1);
    
    % Distanze
    colorMeanDistances = sqrt(sum((meanColors - classMeanColor).^2, 2));
    colorVarDistances = sqrt(sum((varColors - classVarColor).^2, 2));
    
    % Soglie (95° percentile)
    meanThreshold = prctile(colorMeanDistances, 90);
    varThreshold = prctile(colorVarDistances, 90);
    
    % Identifica gli outlier
    meanOutliers = colorMeanDistances > meanThreshold;
    varOutliers = colorVarDistances > varThreshold;
    colorOutliers = meanOutliers | varOutliers;


    %% Combina i filtri
    %distanceThreshold = prctile(distances, 90);
    %varianceThreshold = prctile(variances, 5); % Immagini con varianza troppo bassa
    finalOutliers = distanceThreshold| ...
                    segmentOutliers | ...
                    colorOutliers;

    for i = 1:length(imageFiles)
        if finalOutliers(i)
            movefile(filePaths{i}, outlierFolder);
            fprintf('Immagine spostata nella classe 251 (outlier): %s\n', filePaths{i});
        end
    end
end

fprintf('Rimozione degli outlier completata.\n');
