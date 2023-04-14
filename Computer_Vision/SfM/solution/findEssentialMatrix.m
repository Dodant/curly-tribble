function best_essential = findEssentialMatrix(f1_K, f2_K)

    threshold = 0.00001;
    num_of_inliers = 0;

    % RANSAC
    % iteration = 10000
    for i = 1:10000
        % ramdomly choose 5 indexes
        random_indexes = randsample(length(f1_K),5);
    
        Q1 = f1_K(1:3,random_indexes);
        Q2 = f2_K(1:3,random_indexes);
    
        E_vec = calibrated_fivepoint(Q1,Q2); % 9 x n
        num_of_solution = size(E_vec,2);
        E_mat = reshape(E_vec,3,3,num_of_solution);
        E_mat = permute(E_mat,[2,1,3]); % row major
        
        for j = 1:num_of_solution
            E_f1K = E_mat(:,:,j)*f1_K; % 3 x 3 * 3 x m  = 3 * m
            Et_f2K = E_mat(:,:,j)'*f2_K; % 3 x 3 * 3 x m = 3 * m

            distance = sum(f2_K.*E_f1K).^2 ./ ...
                (E_f1K(1,:).^2 + E_f1K(2,:).^2 + Et_f2K(1,:).^2 + Et_f2K(2,:).^2);
            
            inliers_len = length(find(abs(distance)<threshold));

            if inliers_len > num_of_inliers
                num_of_inliers = inliers_len;
                best_essential = E_mat(:,:,j);
            end
        end
    end