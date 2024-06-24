close all
clc
clear all

% Directory of the .mat files and mask file
folderPath = './mat/';
maskFilePath = 'mask.png'; % Replace with your actual mask file path
files = dir(fullfile(folderPath, '*.mat'));

% Load and prepare the mask
mask = imread(maskFilePath);
mask = im2bw(mask); % Convert mask to binary image if it's not already

% Processing each .mat file
for k = 1:length(files)
    baseFileName = files(k).name;
    fullFileName = fullfile(folderPath, baseFileName);
    
    % Load .mat file
    data = load(fullFileName);
    
    % Get the names of variables in the .mat file
    variableNames = fieldnames(data);
    
    % Assuming the first variable is the one we need
    if ~isempty(variableNames)
        variableName = variableNames{1}; % Get the first variable name
        imageData = squeeze(data.(variableName));
        
        % Resize mask to match imageData size
        resizedMask = imresize(mask, size(imageData));
        
        % Apply mask: Set outside of mask to zero
        maskedData = imageData .* resizedMask;
        
        % Normalize the intensity values within the mask
        maskedData(resizedMask) = mat2gray(maskedData(resizedMask));
        
        % Create an RGB image to apply the hot colormap
        figure;
        imagesc(maskedData);
        colormap hot;
        caxis([0, 1]); % Set color axis scaling to the range of normalized data
        colorbar;
        axis tight;
        axis off;
        
        % Capture the image from the axes
        frame = getframe(gca);
        img = frame2im(frame);
        img = im2uint8(img);
        close(gcf); % Close the figure window

        % Save the image as .jpg
        [filePath, name, ~] = fileparts(fullFileName); % Extract file parts
        jpgFileName = fullfile(filePath, [name, '.jpg']);
        imwrite(img, jpgFileName);
    else
        fprintf('No variables found in file %s\n', baseFileName);
    end
end

