% Percorsi
csvFile = 'train_small.csv'; % File CSV con le etichette
imagesFolder = '../train_set'; % Cartella con tutte le immagini
outputFolder = '../output'; % Cartella di output organizzata
outlierFolder = fullfile(outputFolder, '251'); % Cartella per gli outlier
csvFile_test = 'val_info.csv';
testFolder = '../val_set';

% Leggi il CSV
data = readtable(".." + "/Annotazioni/" + csvFile, 'ReadVariableNames', false);
%Nomi immagini
imageNames = data.Var1;
%etichette
classLabels = data.Var2;
uniqueClasses = unique(classLabels);

% Leggi il CSV Test
data = readtable(".." + "/Annotazioni/" + csvFile_test, 'ReadVariableNames', false);
%Nomi immagini
imageNames_test = data.Var1;
%etichette
classLabels_test = data.Var2;



%DataAugmentation;
%% creazione griglia ogni 'featstep' pixel
disp('Creazione griglia KeyPoints');
imsize=224;
featStep = 15;
pointPositions = [];


disp("creazione griglia ogni 'featstep' pixel");
for ii =featStep: featStep : imsize-featStep
    for jj = featStep : featStep : imsize-featStep
        pointPositions = [pointPositions; ii jj];
    end
end

%% estrazione features sul training

% disp("estrazione features");
% features= [];
% labels=[];
% filePaths = {};
% 
% for idxClass= 1:length(uniqueClasses)
%     classFolder = fullfile(outputFolder , num2str(uniqueClasses(idxClass)));
%     imageFiles = dir(fullfile(classFolder , "*.jpg"));
%     disp("processamento classe " + num2str(idxClass));
%     for ii=1:length(imageFiles)
%         imgPath = fullfile(classFolder , imageFiles(ii).name);
%         img = imread(imgPath);
%         img = rgb2gray(img); %le feature surf sono basate sui gradienti e non sulla feature colore, si riduce numero dati
%         [imFeatures , dontcare] = extractFeatures(img , pointPositions , 'Method','SURF');
%         features = [features ; imFeatures];
%         labels =  [labels ; repmat(idxClass , size(imFeatures,1),1) ...
%             repmat(ii , size(imFeatures,1),1)];
%     end
% end

%save('features.mat', 'features');
%dataFeatures=load("features.mat");
%save("labels.mat" , 'labels');
%features = dataFeatures.features;
%dataLabels = load("labels.mat");
%labels = dataLabels.labels;

%% k-means
disp("clusterizzazione con k-means");
k=1000;

tic
[idx , C] = kmeans(features , k , 'MaxIter', 1200, 'Replicates', 3, 'Start', 'plus');
toc

save('kmeans_results.mat', 'idx', 'C');
%load('kmeans_results.mat');  % Carica idx e C direttamente in workspace

%% istogrammi di training

disp("istogrammi di training");

BOW_tr = [];
labels_tr = [];

for idxClass = 1: length(uniqueClasses)
    classFolder = fullfile(outputFolder , num2str(uniqueClasses(idxClass)));
    imageFiles = dir(fullfile(classFolder , "*.jpg"));
    for ii=1:length(imageFiles)
       u = find(labels(:, 1) == idxClass & labels(:,2) == ii);
       imfeaturesIDX = idx(u);
       H=hist(imfeaturesIDX,1:k);
       H = H/sum(H); %per nostro caso inutile, ma di solito viene fatto perche in immagini dimensioni o numero di punti possono essere diverse. in nostro caso si ha controllo di tutto
       BOW_tr = [BOW_tr;H];
       labels_tr = [labels_tr;idxClass];
    end

end

%% test set: istogrammi

BOW_te = [];
labels_te = [];

for idxImage = 1: 1000
    classFolder = fullfile(testFolder);
    imageFiles = dir(fullfile(testFolder , "*.jpg"));
    imgPath = fullfile(classFolder , imageFiles(idxImage).name);
    img = imread(imgPath);
    img = im2double(img);
    img = imresize(img , [imsize imsize]);
    img = rgb2gray(img);
    [imfeatures  ,dontcare] = extractFeatures(img , pointPositions, 'Method' , "SURF"); 
    
    D = pdist2(imfeatures , C);
    [dontcare , words] = min(D , [] , 2);
    
    H = hist(words , 1:k);
    H = H./sum(H);

    BOW_te = [BOW_te; H];
    labels_te = [labels_te; classLabels_test(idxImage)];

end


% disp('Primi 5 istogrammi di test:')
% disp(BOW_te(1:5, :))
% 
% disp(words(1:10));
% 
% disp(length(words));
%% classificazione test

disp('classificazione test set')

predicted_class=[];
tic

for ii=1:size(BOW_te ,1)
    H = BOW_te(ii , :);
    DH = pdist2(H , BOW_tr);
    u = find(DH == min(DH));
    u = u(1);
    predicted_class = [predicted_class ; labels_tr(u)];
end
toc




%% misurazione performance:

CM = confusionmat(labels_te , predicted_class);
CM = CM./repmat(sum(CM,2),1,size(CM,2));

accuracy = mean(diag(CM));

figure(1), clf
imagesc(CM), colorbar
title(['Accuracy: ' num2str(accuracy)])