function possible_solutions = decompose(E)
    diag = [ 1 0 0; 0 1 0; 0 0 0 ];
    W = [ 0 -1 0; 1 0 0; 0 0 1 ];
    
    [U,~,V] = svd(E); % all 3 x 3
    [U,~,V] = svd(U*diag*V');

    % four possible reconstruction
    possible_solutions = {[U*W*V' U(:,3)], ... 
                          [U*W*V' -U(:,3)], ...
                          [U*W'*V' U(:,3)], ...
                          [U*W'*V' -U(:,3)]};
end