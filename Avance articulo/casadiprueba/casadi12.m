
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clc; clear; close all
import casadi.*

%En esta etapa se integró el controlador predictivo (MPC) con el modelo físico del balance hídrico para generar de manera automática un conjunto de datos sintéticos bajo condiciones operativas realistas. 
% Para ello, el MPC calcula en cada instante la acción óptima de riego utilizando el modelo hidrológico, mientras que la "planta real" se simula como una versión perturbada del mismo modelo, 
% incorporando dinámicas no modeladas y ruido. La diferencia entre la respuesta de la planta real simulada y la del modelo físico define un término residual, que se almacena junto con las variables de estado, 
% control y perturbaciones climáticas. Este procedimiento permite construir un dataset coherente para entrenar posteriormente una red neuronal residual, 
% sentando las bases de un modelo grey-box que combine conocimiento físico y aprendizaje automático.

%% ===============================
% PARÁMETROS FÍSICOS
%% ===============================
Zr = 0.30;       % m
dt = 1;          % día

theta_wp = 0.12;
theta_fc = 0.25;
theta_ref = 0.25;

%% ===============================
% MPC
%% ===============================
N = 10;
Tsim = 200;

%% ===============================
% MODELO CASADI
%% ===============================
x = SX.sym('x');
u = SX.sym('u');
p = SX.sym('p',2);   % [P; ETc]

P_rain = p(1);
ETc    = p(2);

% Stress hídrico
Ks = fmin(1, fmax(0,(x-theta_wp)/(theta_fc-theta_wp)));
ETc_eff = Ks*ETc;

% drenaje
D = fmax(0, x-theta_fc)*5;

x_next = x + dt*(1/(Zr*1000))*(u + P_rain - ETc_eff - D);
f = Function('f',{x,u,p},{x_next});

%% ===============================
% VARIABLES MPC
%% ===============================
X = SX.sym('X',N+1);
U = SX.sym('U',N);

X0 = SX.sym('X0');
P_h = SX.sym('P_h',N);
ETc_h = SX.sym('ETc_h',N);

J = 0; g = [];
g = [g; X(1)-X0];

for k=1:N
    e = X(k)-theta_ref;
    J = J + 10*e^2 + 0.01*U(k)^2;
    g = [g; X(k+1)-f(X(k),U(k),[P_h(k);ETc_h(k)])];
end

OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0;P_h;ETc_h]);
solver = nlpsol('solver','ipopt',OPT);

%% ===============================
% LÍMITES
%% ===============================
lbg = zeros(size(g));
ubg = zeros(size(g));

lbw = [0.1*ones(N+1,1); zeros(N,1)];
ubw = [0.4*ones(N+1,1); 6*ones(N,1)];

%% ===============================
% SIMULACIÓN Y GENERACIÓN DE DATASET
%% ===============================
theta_k = 0.18;
w0 = zeros((N+1)+N,1);

Xdata = zeros(Tsim,4);
Ydata = zeros(Tsim,1);

theta_model_hist = zeros(Tsim,1);
theta_real_hist  = zeros(Tsim,1);
u_hist = zeros(Tsim,1);

for t=1:Tsim

    % perturbaciones climáticas
    ETc_t = 4 + 0.5*sin(2*pi*t/40);
    P_t   = (t>40 & t<60)*4 + (t>120 & t<140)*3;

    P_hor  = P_t*ones(N,1);
    ETc_hor= ETc_t*ones(N,1);

    sol = solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,...
                 'p',[theta_k;P_hor;ETc_hor]);

    w_opt = full(sol.x);
    Uopt = w_opt(N+2:end);
    u_k = Uopt(1);

    % MODELO
    theta_model = full(f(theta_k,u_k,[P_t;ETc_t]));

    % PLANTA REAL (desconocida)
    residual = 0.02*sin(5*theta_k) + 0.01*theta_k.^2 + 0.005*randn;
    theta_real = theta_model + residual;

    % GUARDAR DATASET
    Xdata(t,:) = [theta_k u_k P_t ETc_t];
    Ydata(t) = theta_real - theta_model;

    % cerrar lazo
    theta_k = theta_real;

    % logs
    theta_model_hist(t)=theta_model;
    theta_real_hist(t)=theta_real;
    u_hist(t)=u_k;

    w0 = w_opt;
end

%% ===============================
% GRÁFICAS
%% ===============================
t_vec = 1:Tsim;
fs = 18;          % Tamaño de letra maestro coincidente
fs_tit = fs + 4;  
negro = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Humedad (Modelo vs Real) ---
subplot(3,1,1)
plot(t_vec, theta_model_hist, 'b', 'LineWidth', 3); hold on;
plot(t_vec, theta_real_hist, 'r--', 'LineWidth', 2.5);

% Etiquetas de límites (Misma posición que el definitivo)
yline(theta_wp, 'k:', 'Punto Marchitez', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
yline(0.40, 'k:', 'Saturación', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');

% Ajuste de escala para legibilidad
ylim([0.05, 0.55]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del suelo (Generación de Dataset)', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;

set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5, 'FontWeight', 'bold');

lgdd = legend('Modelo', 'Planta Real', 'Location', 'northeast');
lgdd.FontSize = fs-2;
lgdd.TextColor = negro;
lgdd.Color = 'w';
lgdd.EdgeColor = negro;

% --- Subplot 2: Acción de Control (Riego) ---
subplot(3,1,2)
stairs(t_vec, u_hist, 'k', 'LineWidth', 3); hold on;

% Límite de riego
yline(6.0, 'r:', 'Máx. Riego (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold', ...
    'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');

ylim([-0.5, 9]); 
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de Control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;

set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5, 'FontWeight', 'bold');

% --- Subplot 3: Residuo Real (Objetivo NN) ---
subplot(3,1,3)
plot(t_vec, Ydata, 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 3); % Color verde para el residuo

ylim([min(Ydata)-0.05, max(Ydata)+0.05]); 
ylabel('\Delta\theta (Residuo)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Residuo real (Objetivo de la Red Neuronal)', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;

set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5, 'FontWeight', 'bold');

%% ===============================
% GUARDAR DATASET
%% ===============================
save greybox_dataset.mat Xdata Ydata
disp('Dataset guardado: greybox_dataset.mat')
