function X = get3DPoint(possible_reconstructions, f1_K, f2_K)
    T = 1.5;
    pr1 = eye(3,4);
    X_list = cell(1,4);
    depth_score = [];

    for i = 1:4
        pr2 = cell2mat(possible_reconstructions(i));
        tmp = [];

        for j = 1:length(f1_K)
            % A = 4 x 1
            A = [f1_K(1,j) * pr1(3,:) - pr1(1,:);
                 f1_K(2,j) * pr1(3,:) - pr1(2,:);
                 f2_K(1,j) * pr2(3,:) - pr2(1,:);
                 f2_K(2,j) * pr2(3,:) - pr2(2,:);];
            [~,~,V] = svd(A); % V = 4 x 4
            tmp(:, j) = V(:,4); % temp = 4 x m
        end

        tmp = tmp./tmp(4,:);
        X = tmp(1:3,:); % X = 3 x m
        X_list(i) = {X};

        t = -det([pr2(:,1),pr2(:,2),pr2(:,3)]);
        center = [det([pr2(:,2),pr2(:,3),pr2(:,4)])/t; 
                  -det([pr2(:,1),pr2(:,3),pr2(:,4)])/t; 
                   det([pr2(:,1),pr2(:,2),pr2(:,4)])/t];

        for j = 1:length(f1_K)
            depth1(j) = X(3,j) / T;
            w = pr2(3,1:3) * (X(1:3,j) - center(1:3,:));
            depth2(j) = (sign(det(pr2(:,1:3))) * w) /...
                        (T * norm(pr2(3,1:3)));
        end
        depth_score(i) = sum(sign(depth1)+sign(depth2));
    end
    disp(depth_score)
    [~, arg] = max(depth_score);
    disp(['Solution Index - ', int2str(arg)])
    X = cell2mat(X_list(arg));
end