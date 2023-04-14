run('VLROOT/toolbox/vl_setup.m')

% 1. Load the input images
disp('1 - Load the input images');
filename1 = 'two_view/sfm01.jpg';
filename2 = 'two_view/sfm02.jpg';
% Best Pair - (13, 25)
% filename1 = 'multi_view/13.jpg';
% filename2 = 'multi_view/25.jpg';
img1 = single(rgb2gray(imread(filename1))); % 1366 x 2048 single
img2 = single(rgb2gray(imread(filename2))); % 1366 x 2048 single


% 2. Extract feature from both images
% SIFT
% feature - 4 x K matrix (x1, x2, sigma, theta)
% descriptors - 128 x K matrix (128bit demension)
disp('2 - Extract feature from both images')
[F1,D1] = vl_sift(img1,'PeakThresh',1);
[F2,D2] = vl_sift(img2,'PeakThresh',1);


% 3. Match features between two images
disp('3 - Match features')
[matches,~] = vl_ubcmatch(D1,D2); %  matches = 2 x K

match_f1 = zeros(2,length(matches)); % 2 x 270
match_f2 = zeros(2,length(matches)); % 2 x 270
    
for i = 1:length(matches)
    match_f1(1:2,i) = F1(1:2,matches(1,i)); % F1 x,y
    match_f2(1:2,i) = F2(1:2,matches(2,i)); % F2 x,y
end

K = [ 1506.070 0 1021.043; 0 1512.965 698.031; 0 0 1 ];
f1_K = K\[match_f1; ones(1,length(match_f1))]; % 3 x 270
f2_K = K\[match_f2; ones(1,length(match_f2))]; % 3 x 270


% 4. Estimate Essential Matrix E with RANSAC
% intrinsic matrix K = 3 x 3
disp('4 - Estimate Essential Matrix E')
E = findEssentialMatrix(f1_K,f2_K);
disp('- - Best Essential'); disp(E);


% 5. Decompose E to camera extrinsic[R|T]
disp('5 - Decompose E')
diag = [ 1 0 0; 0 1 0; 0 0 0 ];
W = [ 0 -1 0; 1 0 0; 0 0 1 ];

[U,~,V] = svd(E); % all 3 x 3
[U,~,V] = svd(U*diag*V');

% four possible reconstruction
possible_reconstructions = {[U*W*V' U(:,3)], ... 
                            [U*W*V' -U(:,3)], ...
                            [U*W'*V' U(:,3)], ...
                            [U*W'*V' -U(:,3)]};


% 6. Generate 3D point by implementing triangulation
disp('6 - Generate 3D point')
X = get3DPoint(possible_reconstructions,f1_K,f2_K); % X - 3 x m 

% plot X
figure; pcshow(X'); shg;
xlabel('x'); ylabel('y'); zlabel('z')
axis([-10 10 -10 10 -10 10]); shg;

% Save 3d Points
disp('Save 3d Points')
SavePLY('two_view_sfm.ply', X);
