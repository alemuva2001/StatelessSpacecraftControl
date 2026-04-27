function [A_BN] = wahbaProblem(ref, current, camera)

% Performs the computations to solve the wahba problem
% -ref: vector that contains the image coordinates of the stars in the reference measure 
% -current: vector that contains the image coordinates of the stars in the actual measure
% -camera: struct that contains the information of the camera
%   ·f: focal distance
%   ·W: Width of the image
%   ·H: Height of the image

%Camera parameters
f = camera.f;
cx = camera.W;
cy = camera.H;

rel = current;

% Tridimensional star vector
ref_3D = 1/(sqrt(dot(ref(1,:)-cx,ref(1,:)'-cx)+dot(ref(2,:)-cy, ref(2,:)'-cy)^2+f^2))*[ref(1,:)-cx; ref(2,:)-cy; ones(1,length(ref(1,:)))*f];
rel_3D = 1/(sqrt(dot(rel(1,:)-cx,rel(1,:)'-cx)+dot(rel(2,:)-cy, rel(2,:)'-cy)^2+f^2))*[rel(1,:)-cx; rel(2,:)-cy; ones(1,length(rel(1,:)))*f];

%% Wahba Problem Resolution
% Matriz de perfil
H = ref_3D*rel_3D';

% SVD
[U, ~, V] = svd(H);
flip = det(U)*det(V');
U(:,size(U,2)) = U(:,size(U,2))*flip;

% Rotation matrix from the reference to the current attitude
A_BN = U*V';

end