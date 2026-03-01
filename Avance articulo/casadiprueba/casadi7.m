%% =========================
% MPC con acción integral
% =========================

addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');


clear; clc;
import casadi.*

%Se introdujo un estado integral del error de humedad para eliminar el error en régimen permanente observado en el MPC original. 
% Esto garantiza seguimiento exacto de la referencia, incluso ante perturbaciones constantes."
%Inicialmente el MPC presentaba error estacionario debido a pérdidas constantes por evapotranspiración y drenaje. 
% Para resolverlo, se incorporó acción integral en la formulación del MPC, lo que permitió eliminar el error permanente y 
% lograr seguimiento exacto de la referencia de humedad, incluso en presencia de perturbaciones climáticas.

%% Parámetros
dt = 1;               % paso de tiempo (día)
N  = 10;              % horizonte MPC
theta_ref = 0.25;     % referencia de humedad

ETc_mean = 4;         % ETc promedio
P_mean   = 0;         % precipitación promedio
D_const  = 1.0;       % drenaje constante (por ahora)

%% Estados simbólicos
theta = SX.sym('theta');   % humedad del suelo
z     = SX.sym('z');       % estado integral

x = [theta; z];

%% Control
u = SX.sym('u');           % riego

%% Dinámica del sistema
theta_dot = u + P_mean - ETc_mean - D_const;
theta_next = theta + dt*theta_dot;

z_next = z + (theta - theta_ref);

x_next = [theta_next; z_next];

f = Function('f',{x,u},{x_next});

%% Variables de optimización
U = SX.sym('U',N);           % controles
X = SX.sym('X',2,N+1);       % estados

P = SX.sym('P',2);           % estado inicial

%% Función de costo
Q_theta = 100;
Q_z     = 10;
R_u     = 0.1;

obj = 0;
g   = [];

g = [g; X(:,1) - P];

for k = 1:N
    xk = X(:,k);
    uk = U(k);

    obj = obj + ...
        Q_theta*(xk(1)-theta_ref)^2 + ...
        Q_z*(xk(2))^2 + ...
        R_u*(uk)^2;

    x_next_pred = f(xk,uk);
    g = [g; X(:,k+1) - x_next_pred];
end

OPT_variables = [reshape(X, 2*(N+1), 1); U];

nlp = struct('f',obj,'x',OPT_variables,'g',g,'p',P);

opts.ipopt.print_level = 0;
solver = nlpsol('solver','ipopt',nlp,opts);

%% Simulación
Tsim = 30;
theta_hist = zeros(Tsim,1);
z_hist     = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);

x0 = [0.15; 0];   % humedad inicial + integral

for t = 1:Tsim

    lbx = -inf*ones(size(OPT_variables));
    ubx = inf*ones(size(OPT_variables));

    lbg = zeros(size(g));
    ubg = zeros(size(g));

    sol = solver('x0',zeros(size(OPT_variables)),...
                 'lbx',lbx,'ubx',ubx,...
                 'lbg',lbg,'ubg',ubg,...
                 'p',x0);

    sol_x = full(sol.x);
    u_opt = sol_x(end-N+1);

    x0 = full(f(x0,u_opt));

    theta_hist(t) = x0(1);
    z_hist(t)     = x0(2);
    u_hist(t)     = u_opt;
end

%% Gráficas
figure;
subplot(2,1,1)
plot(theta_hist,'LineWidth',2); hold on
yline(theta_ref,'--r')
ylabel('\theta')
title('Humedad del suelo')

subplot(2,1,2)
stairs(u_hist,'LineWidth',2)
ylabel('Riego')
xlabel('Tiempo (días)')
title('Acción de control')



