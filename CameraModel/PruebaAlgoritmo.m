clc;
clear all;
%close all;

%Carga de datos
R_BN_1 = load("R_BN_1.mat","R_BN_1");
u_1 = load("u_1.mat","u_1");
v_1 = load("v_1.mat","v_1");

%Sensor Parameters
FOV = deg2rad(60);

W = 1024; %Horizontal resolution [px]
H = 1024; %Vertical resolution [px]

f = W/(2*tan(FOV/2)); %Focal distance

cx = W/2;
cy = H/2;

% Vector de puntos inicial
u = u_1.u_1;
v = v_1.v_1;

ref = [u v]';

% Vector tridimensional de estrellas
ref_3D = 1/(sqrt(dot(ref(1,:)-cx,ref(1,:)'-cx)+dot(ref(2,:)-cy, ref(2,:)'-cy)^2+f^2))*[ref(1,:)-cx; ref(2,:)-cy; ones(1,length(ref(1,:)))*f];

%Rotacion de los puntos
roll = deg2rad(10);
pitch = deg2rad(10);
yaw = deg2rad(10);

R = eul2rotm([roll pitch yaw]);

rel_3D = R*ref_3D;
%rel_3D = 1/(sqrt(dot(ref(1,:)-cx,ref(1,:)'-cx)+dot(ref(2,:)-cy, ref(2,:)'-cy)^2+f^2))*[ref(1,:)-cx; ref(2,:)-cy; ones(1,length(ref(1,:)))*f];


%Visualizacion
figure()
subplot(1,2,1)
plot(ref_3D(1,:),ref_3D(2,:),'LineStyle','none', 'Marker','*','MarkerSize',10)
hold on
plot(rel_3D(1,:),rel_3D(2,:),'LineStyle','none', 'Marker','*','MarkerSize',10)
grid on;
%axis([0 W 0 H]);
set(gca, 'YDir', 'reverse');
daspect([1 1 1])
legend('Reference', 'Rotate','Location','bestoutside')

%% Calculo de la rotacion mediante Wahba

% Matriz de perfil
H = ref_3D*rel_3D';

% SVD
[U, S, V] = svd(H);
flip = det(U)*det(V');
U(:,size(U,2)) = U(:,size(U,2))*flip;

% Matriz de Rotacion
R_pred = U*V';

pred = R_pred*rel_3D;

%Comprobacion de Resultados
subplot(1,2,2)
plot(ref_3D(1,:),ref_3D(2,:),'LineStyle','none', 'Marker','*','MarkerSize',10)
hold on
plot(rel_3D(1,:),rel_3D(2,:),'LineStyle','none', 'Marker','*','MarkerSize',10)
plot(pred(1,:),pred(2,:),'LineStyle','none', 'Marker','*','MarkerSize',10)
grid on;
legend('Reference', 'Rotated','Predicted','Location','bestoutside')
set(gca, 'YDir', 'reverse');
daspect([1 1 1])


%%

ref = [u(:,1) v(:,1)];

rel = [u(:,50) v(:,50)];

D = pdist2(ref,rel);

[d,i]=min(D);

figure()
plot(ref(:,1), ref(:,2),'LineStyle','none','Marker','*','MarkerSize',10)
hold on
plot(rel(:,1), rel(:,2),'LineStyle','none','Marker','*','MarkerSize',10)
set(gca, 'YDir', 'reverse');
axis([0,W,0,H])
daspect([1 1 1])