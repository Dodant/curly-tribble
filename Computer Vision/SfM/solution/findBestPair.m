num_of_matches = 0;
pair1 = 0;
pair2 = 0;


for i=1:25
    filename1 = ['multi_view/' int2str(i) '.jpg'];
    for j=1:25
        if i == j
            continue
        end
        filename2 = ['multi_view/' int2str(j) '.jpg'];
        img1 = single(rgb2gray(imread(filename1)));
        img2 = single(rgb2gray(imread(filename2)));
        [F1, D1] = vl_sift(img1, 'PeakThresh', 3);
        [F2, D2] = vl_sift(img2, 'PeakThresh', 3);
        [matches, score] = vl_ubcmatch(D1,D2); 
        disp([int2str(i), ' ', int2str(j), ' - ', int2str(size(matches, 2))]);
        if num_of_matches <= size(matches, 2)
            num_of_matches = size(matches, 2);
            pair1 = i;
            pair2 = j;
        end
    end
end
disp(['Num of Matches - ' int2str(num_of_matches)]);
disp(['Best Pair - ', int2str(pair1), ' ', int2str(pair2)]);

% Best Pair
% 13 - 25