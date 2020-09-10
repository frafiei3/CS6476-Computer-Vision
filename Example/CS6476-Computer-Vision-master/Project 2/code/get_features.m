% Local Feature Stencil Code
% CS 4495 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

features = zeros(size(x,1), 128);

for i = 1:size(x,1)
    feature = zeros(1,128);
    
    block = image(y(i)-7:y(i)+8,x(i)-7:x(i)+8);
    [Gmag, Gdir] = imgradient(block);
    
    gaussian_filter = fspecial('gauss',[16,16], 4);
    weighted_Gmag = Gmag.*gaussian_filter; 
    
    dvd_mag = mat2cell(weighted_Gmag,4*ones(1,4),4*ones(1,4));
    dvd_ori = mat2cell(Gdir,4*ones(1,4),4*ones(1,4));
    
    for row = 1:4
        for col = 1:4
            for m = 1:4
                for n = 1:4
                    bin = ceil((dvd_ori{row,col}(m,n)+180)/45);
                    if bin == 0
                        bin = 1;
                    end
                    feature(((row-1)*4+col-1)*8+bin) = feature(((row-1)*4+col-1)*8+bin) + dvd_mag{row,col}(m,n);
                end
            end
        end
    end
    
    feature = feature ./ norm(feature);
    feature(feature(:)>0.2) = 0.2;
    feature = feature ./ norm(feature);

    features(i,:) = feature;
    
end

end






