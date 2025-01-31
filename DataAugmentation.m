PuliziaDati;

% obiettivo: effettuare data augmentation per le 7 immagini più vicine al
% centroide del cluster, in modo da aumentare il dataset con le immagini di
% cui si ha maggiore sicurezza che non siano outlier, che potrebbero ancora
% essere presenti in seguito alla pulizia effettuata nello script
% PuliziaDati.m

nAugmentations = 3;
nImAugmented=5;

for classIdx = 1:length(uniqueClasses)
    classFolder = fullfile(outputFolder , num2str(uniqueClasses(classIdx)));
    imageFiles = dir(fullfile(classFolder , "*.jpg"));
    features = [];
    filePaths = {};
    for i=1:length(imageFiles)
        imgPath = fullfile(classFolder , imageFiles(i).name);
        img = imread(imgPath);
        img = imresize(img , [224 , 224]);

        feature = activations(net , img , layer , 'OutputAs' , 'rows');
        features = [features ; feature];

        filePaths{end+1} = imgPath; 
    end

    %clusterizzazione con K-means
    [~ , C] = kmeans(features, 1, 'Replicates',5 , 'Distance','sqeuclidean');
    distances = pdist2(features , C); %calcolo distanze centroide

    %ordina le immagini in base alla distanza dal centroide
    [~ , sortedIndices] = sort(distances , 'ascend'); %distanza crescente
    closestImages = sortedIndices(1:min(nImAugmented, length(imageFiles))); % prende il numero di immagini inferiore tra le immagini scelte e il numero di immagini presenti della classe

    className = num2str(uniqueClasses(classIdx));
    fprintf('Data Augmentation per la classe %s... \n' , className);
        for idx = 1:length(closestImages)
            imgPath = filePaths{closestImages(idx)};
            img = imread(imgPath);
            %estrazione nome ed estensione file originale da aumentare
            [ignore , name , ext]= fileparts(imgPath);

            %applica data augmentation
            for augIdx=1:nAugmentations
                augmentedImg = img;
                %rotazione
                angle = randi([-20, 20]);
                augmentedImg = imrotate(augmentedImg, angle , 'crop');

                %zoom Casuale
                scale = 1+ (rand() * 0.2 - 0.1);
                augmentedImg = imresize(augmentedImg , scale);
                augmentedImg = imresize(augmentedImg , [224,224]);

                %luminosità
                brightnessFactor = 0.6 + rand() * 0.8;
                augmentedImg = augmentedImg * brightnessFactor;
                augmentedImg = uint8(min(255, augmentedImg)); % Clip ai valori validi
                
                % Flip orizzontale (probabilità 50%)
                if rand() > 0.5
                    augmentedImg = flip(augmentedImg, 2);
                end
                
                % Aggiunta di rumore
                if rand() > 0.8
                augmentedImg = imnoise(augmentedImg, 'gaussian', 0, 0.01);
                end

                %genera nome per immagine generata
                augmentedName = sprintf('%s_aug%d%s' , name, augIdx, ext);
                outputPath = fullfile(classFolder , augmentedName);

                imwrite(augmentedImg, outputPath);
            end
        end
end



