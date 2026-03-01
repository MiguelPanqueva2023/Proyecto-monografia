addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');

clear; clc; close all
import casadi.*

%% ================================
% Parámetros físicos del suelo
% ================================
Zr = 0.30;              % Profundidad radicular [m]
dt = 1;                % Paso de tiempo [día]
theta_fc = 0.30;       % Capacidad de campo
theta_wp = 0.12;       % Punto de marchitez
theta_ref = 0.25;      % Humedad objetivo

Dmax = 4;              % drenaje máximo [mm/día]

%% ================================
% Horizonte MPC
% ================================
N = 10;

%% ================================
% Variables simbólicas
% ================================
x = SX.sym('x');      % humedad del suelo
u = SX.sym('u');      % riego
P = SX.sym('P');      % precipitación
ETp = SX.sym('ETp');  % ETc,pot

%% ================================
% Estrés hídrico Ks(theta)
% ================================
Ks = (x - theta_wp)/(theta_fc - theta_wp);
Ks = fmax(0,fmin(1,Ks));

%% ================================
% Drenaje dependiente de humedad
% ================================
D = Dmax*fmax(0,(x-theta_fc)/(1-theta_fc));

%% ================================
% Dinámica del sistema
% ================================
x_next = x + dt*(1/Zr*1000)*( u + P - ETp*Ks - D );
f = Function('f',{x,u,P,ETp},{x_next});

%% ================================
% Variables de optimización
% ================================
X = SX.sym('X',N+1);
U = SX.sym('U',N);
Pp = SX.sym('Pp',N);
ETp_p = SX.sym('ETp_p',N);
X0 = SX.sym('X0');

%% ================================
% Costo y restricciones
% ================================
J = 0;
g = [X(1)-X0];

for k=1:N
    J = J + (X(k)-theta_ref)^2 + 0.01*U(k)^2;
    g = [g; X(k+1) - f(X(k),U(k),Pp(k),ETp_p(k))];
end

OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0;Pp;ETp_p]);
solver = nlpsol('solver','ipopt',OPT);

%% ================================
% Límites
% ================================
lbg = zeros(N+1,1); ubg = lbg;
lbw = [0.10*ones(N+1,1); zeros(N,1)];
ubw = [0.40*ones(N+1,1); 6*ones(N,1)];

%% ================================
% SIMULACIÓN
% ================================
Tsim = 30;
theta = 0.18;

theta_hist=zeros(Tsim,1);
u_hist=zeros(Tsim,1);
ET_hist=zeros(Tsim,1);
P_hist=zeros(Tsim,1);

w0=zeros((N+1)+N,1);

%% ================================
% Escenarios climáticos
% ================================
t=1:Tsim;
ETc_pot = 4 + 1.2*sin(2*pi*t/30);      % ET climática
Pclim = zeros(1,Tsim);
Pclim(10:13)=4;                       % tormenta
Pclim(20:23)=3;

for k=1:Tsim

    Pp_val = Pclim(min(k:k+N-1,Tsim));
    ETp_val = ETc_pot(min(k:k+N-1,Tsim));

    sol = solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg, ...
                 'p',[theta;Pp_val(:);ETp_val(:)]);

    w = full(sol.x);
    Uopt = w(N+2:end);
    uk = Uopt(1);

    theta = full(f(theta,uk,Pclim(k),ETc_pot(k)));

    theta_hist(k)=theta;
    u_hist(k)=uk;
    ET_hist(k)=ETc_pot(k);
    P_hist(k)=Pclim(k);

    w0 = [w(2:N+1);w(N+1);Uopt(2:end);Uopt(end)];
end

%% ================================
% Gráficas
% ================================
figure
subplot(3,1,1)
plot(theta_hist,'LineWidth',2), hold on
yline(theta_ref,'r--'), title('Humedad del suelo')

subplot(3,1,2)
stairs(u_hist,'LineWidth',2), title('Riego MPC')

subplot(3,1,3)
plot(ET_hist,'b'), hold on
stairs(P_hist,'r'), title('Perturbaciones climáticas')
legend('ETc,pot','Precipitación')
