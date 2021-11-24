function X = get3DPoint(cases, f1_K, f2_K)
    % CP = Camera Pose
    T = 1.5;
    cp1 = eye(3,4);
    X_list = cell(1,4);
    depth_score = [];

    for i = 1:4
        cp2 = cell2mat(cases(i));
        tmp = [];

        for j = 1:length(f1_K)
            % A = 4 x 1
            A = [f1_K(1,j) * cp1(3,:) - cp1(1,:);
                 f1_K(2,j) * cp1(3,:) - cp1(2,:);
                 f2_K(1,j) * cp2(3,:) - cp2(1,:);
                 f2_K(2,j) * cp2(3,:) - cp2(2,:);];
            [~,~,V] = svd(A); % V = 4 x 4
            tmp(:, j) = V(:,4); % temp = 4 x m
        end

        tmp = tmp./tmp(4,:);
        X = tmp(1:3,:); % X = 3 x m
        X_list(i) = {X};
        
        cp2 = cell2mat(cases(i));
        t = -det([cp2(:,1),cp2(:,2),cp2(:,3)]);
        center2 = [det([cp2(:,2),cp2(:,3),cp2(:,4)])/t; 
                  -det([cp2(:,1),cp2(:,3),cp2(:,4)])/t; 
                   det([cp2(:,1),cp2(:,2),cp2(:,4)])/t];

        for j = 1:length(f1_K)
            depth1(j) = X(3,j) / T;
            w = cp2(3,1:3) * (X(1:3,j) - center2(1:3,:));
            depth2(j) = (sign(det(cp2(:,1:3))) * w) /...
                            (T * norm(cp2(3,1:3)));
        end
        depth_score(i) = sum(sign(depth1)+sign(depth2));
    end
    [~, arg] = max(depth_score);
    disp(['Solution Index - ', int2str(arg)])
    X = cell2mat(X_list(arg));
end