imds = imageDatastore("Railway")
groundtruth = readtable("rails_edited.csv")
imds.Labels = categorical(groundtruth.labels)
%%
classes = categories(imds.Labels)
%%
imshow(readimage(imds,1))
imshow(readimage(imds,2))
imshow(readimage(imds,3))
%%
[trainimds,testimds] = splitEachLabel(imds,0.8,"randomized")
%%
augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5], ...
    'RandXScale', [0.8, 1.2], ...
    'RandYScale', [0.8, 1.2]);

trainds = augmentedImageDatastore([224 224],trainimds,"DataAugmentation",augmenter)
testds = augmentedImageDatastore([224 224],testimds,"DataAugmentation",augmenter)
%%
net = imagePretrainedNetwork("resnet18",NumClasses=numel(classes))
%%
opts = trainingOptions("adam","Metrics","accuracy","VerboseFrequency",1,"MaxEpochs",15,"InitialLearnRate",0.001,"Plots","training-progress")
[cracknet,info] = trainnet(trainds,net,"crossentropy",opts)
