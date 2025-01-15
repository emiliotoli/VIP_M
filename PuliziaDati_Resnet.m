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

%% Pulizia dati mediante utilizzo di modello preaddestrato ResNet18
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
    end
end
% Salva le feature in un file MAT
save('features.mat', 'features', 'imageFiles');

%load('features.mat', 'features', 'filePaths');
% Ottieni le classi uniche
uniqueClasses = unique(classLabels);

% Itera su ogni classe
for classIdx = 1:length(uniqueClasses)
    % Filtra le feature e i percorsi per la classe corrente
    currentClass = uniqueClasses(classIdx);
    classFeatures = features(classLabels == currentClass, :);
    classFilePaths = filePaths(classLabels == currentClass);
    
    % Esegui K-means con un solo cluster
    k = 1;
    [~, C] = kmeans(classFeatures, k, 'Replicates', 5, 'Distance', 'sqeuclidean');
    
    % Calcola le distanze dal centroide
    distances = pdist2(classFeatures, C);
    
    % Determina una soglia per gli outlier
    threshold = prctile(distances, 95); % 95° percentile
    outliers = distances > threshold;  % Immagini oltre la soglia
    
    % Sposta gli outlier nella cartella di rigetto
    outlierPaths = classFilePaths(outliers);
    fprintf('Classe %d: rilevati %d outlier.\n', currentClass, sum(outliers));
    
    for i = 1:length(outlierPaths)
        movefile(filePaths{i}, outlierFolder);
        fprintf('Immagine spostata nella classe 251 (outlier): %s\n', filePaths{i});
    end
end
