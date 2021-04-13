clear all, close all, clc
%% Generate Data
Beta = [10; 28; 8/3]; % Lorenz's parameters (chaotic)
n = 3;
polyorder=2;
x0=[-8; 8; 27];  % Initial condition
tspan=[.01:.01:50];
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,x]=ode45(@(t,x) lorenz(x,Beta),tspan,x0,options);

%% plot data from solution of ode45:
subplot(2,1,1)
plot3(x(:,1),x(:,2),x(:,3))
axis([-20 20 -20 20 0 50]);
xlabel('x')
ylabel('y')
zlabel('z')
grid on
title('Plot of Data')




%% Add random Gaussian Noise: 
Nois=randn(5000,3)./2;
x=x+Nois;


%% subplots of Noisy data:
subplot(2,1,2)
for i=1:length(x(:,1))-1
    P1=[x(i,1),x(i,2),x(i,3)];
    P2=[x(i+1,1),x(i+1,2),x(i+1,3)];
    pts = [P1; P2];

plot3(pts(:,1), pts(:,2), pts(:,3))
hold on
pause(0)
axis([-20 20 -20 20 0 50]);
xlabel('x')
ylabel('y')
zlabel('z')
grid on
title('Plot of with added Gaussian Noise')
end


%% Compute Derivative
for i=1:length(x)
    dx(i,:) = lorenz(x(i,:),Beta);
end


%% compute derivative using diff. function
% dx = diff(x)./diff(t);
% dx=[dx;dx(end,:)];


% %% compute derivative using Total Variation Differentiation:
% iter=500;
% alph=100;
% derivativ=diff(x);
% u0=derivativ(1,:);
% scale='small';
% ep=1e-6;
% dx=1./(size(x,1));
% outputt = TVRegDiff(x, iter, alph, u0, scale, ep, dx, 1, 1 );

%% Build library and compute sparse regression
Theta = poolData(x,n,polyorder,0);  % up to third order polynomials
lambda = 0.6;      % lambda is our sparsification knob.
Xi = sparsifyDynamics(Theta,dx,lambda,n)
poolDataLIST({'x','y','z'},Xi,n,polyorder,0);


