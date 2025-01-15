%% organizzazione file

% Percorsi
csvFile = 'train_small.csv'; % File CSV con le etichette
imagesFolder = '../train_set'; % Cartella con tutte le immagini
outputFolder = '../output'; % Cartella di output organizzata
outlierFolder = fullfile(outputFolder, '251'); % Cartella per gli outlier

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
layer = 'pool5'; % Strato per feature extraction (corretto per ResNet18)
analyzeNetwork(net);


%% Analisi di coerenza interna per classe: pulizia train set
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
    
    % Calcolo del centroide della classe
    centroid = mean(features, 1);
    distances = vecnorm(features - centroid, 2, 2); % Distanza euclidea
    
    % Segmentazione per identificare aree di cibo (esempio basato su soglia)
    segmentOutliers = false(size(imageFiles));
    for i = 1:length(imageFiles)
        imgPath = fullfile(classFolder, imageFiles(i).name);
        img = imread(imgPath);
        
        % Converti in scala di grigi e binarizza
        grayImg = rgb2gray(img);
        binaryImg = imbinarize(grayImg, 'adaptive', 'Sensitivity', 0.5);
        
        % Controlla area segmentata
        segmentedArea = sum(binaryImg(:));
        totalArea = numel(binaryImg);
        
        % Se l'area segmentata è troppo piccola o troppo grande, è un outlier
        if segmentedArea < 0.1 * totalArea || segmentedArea > 0.9 * totalArea
            segmentOutliers(i) = true;
        end
    end
    
    % Combina i filtri (distanza, varianza, segmentazione)
    distanceThreshold = prctile(distances, 95);
    varianceThreshold = prctile(variances, 5); % Immagini con varianza troppo bassa
    finalOutliers = (distances > distanceThreshold) | ...
                    (variances < varianceThreshold) | ...
                    segmentOutliers;
    
    % Sposta gli outlier nella cartella degli outlier
    for i = 1:length(imageFiles)
        if finalOutliers(i)
            movefile(filePaths{i}, outlierFolder);
            fprintf('Immagine spostata nella classe 251 (outlier): %s\n', filePaths{i});
        end
    end
end

fprintf('Rimozione degli outlier completata.\n');
