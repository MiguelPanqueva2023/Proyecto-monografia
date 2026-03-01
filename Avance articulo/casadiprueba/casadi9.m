%% MPC para balance hídrico del suelo
% Versión con saturaciones físicas duras y no linealidades suaves
% Autor: ---
% Objetivo: validar MPC físico antes de Grey-Box

addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');


clear; clc; close all;
import casadi.*

%Inicialmente se implementaron saturaciones físicas suaves para representar límites hidrológicos de forma diferenciable. 
% Sin embargo, las simulaciones mostraron que estas no garantizan la preservación de estados físicamente admisibles bajo perturbaciones intensas. 
% Por ello, se decidió mantener restricciones duras sobre la humedad del suelo, mientras que las no linealidades suaves se conservaron únicamente en los flujos del modelo.

%% -------------------------
% Parámetros físicos
% -------------------------
Zr = 0.30;        % Profundidad radicular [m]
dt = 1.0;         % Paso de tiempo [día]

theta_ref = 0.25; % Humedad deseada

% Límites físicos del suelo
theta_wp  = 0.15; % Punto de marchitez
theta_fc  = 0.25; % Capacidad de campo
theta_sat = 0.35; % Saturación

%% -------------------------
% Horizonte MPC
% -------------------------
N = 10;           % Horizonte de predicción

%% -------------------------
% Definición simbólica
% -------------------------
x = SX.sym('x');      % Humedad del suelo
u = SX.sym('u');      % Riego
p_ETc = SX.sym('ET'); % ETc(t)
p_P   = SX.sym('P');  % Precipitación(t)

%% -------------------------
% ETc dependiente de theta (suave)
% Ks: coeficiente de estrés hídrico
% -------------------------
Ks = fmin(fmax((x - theta_wp)/(theta_fc - theta_wp), 0), 1);
ETc_eff = Ks * p_ETc;

%% -------------------------
% Drenaje dependiente de theta (suave)
% -------------------------
k_d = 15; % pendiente del drenaje
D_max = 5;

D_theta = D_max ./ (1 + exp(-k_d*(x - theta_fc)));

%% -------------------------
% Modelo dinámico (Euler)
% -------------------------
x_next = x + dt*(1/(Zr*1000))*(u + p_P - ETc_eff - D_theta);

f = Function('f',{x,u,p_ETc,p_P},{x_next});

%% -------------------------
% Variables de optimización
% -------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);

X0 = SX.sym('X0');
ETc_h = SX.sym('ETc_h',N);
P_h   = SX.sym('P_h',N);

%% -------------------------
% Función objetivo y restricciones
% -------------------------
J = 0;
g = [];

% Condición inicial
g = [g; X(1) - X0];

for k = 1:N
    % Función de costo
    J = J + (X(k) - theta_ref)^2 + 0.05*U(k)^2;

    % Dinámica
    g = [g; X(k+1) - f(X(k),U(k),ETc_h(k),P_h(k))];
end

%% -------------------------
% Formulación NLP
% -------------------------
OPT = struct( ...
    'x', [X; U], ...
    'f', J, ...
    'g', g, ...
    'p', [X0; ETc_h; P_h] ...
);

opts.ipopt.print_level = 0;
opts.print_time = false;

solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Restricciones DURAS (físicas)
% -------------------------
lbg = zeros(N+1,1);
ubg = zeros(N+1,1);

lbw = [ ...
    theta_wp*ones(N+1,1);   % theta mínima (DURA)
    0.0*ones(N,1)           % riego mínimo
];

ubw = [ ...
    theta_sat*ones(N+1,1);  % theta máxima (DURA)
    6.0*ones(N,1)           % riego máximo
];

%% -------------------------
% Simulación en lazo cerrado
% -------------------------
Tsim = 30;
xk = 0.18;

theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);
ETc_hist   = zeros(Tsim,1);
P_hist     = zeros(Tsim,1);

w0 = zeros((N+1)+N,1);

for t = 1:Tsim

    % ETc(t) suave
    ETc_t = 4 + 1*sin(2*pi*t/30);

    % Precipitación por eventos
    P_t = 0;
    if (t >= 10 && t <= 12) || (t >= 20 && t <= 22)
        P_t = 4;
    end

    ETc_pred = ETc_t*ones(N,1);
    P_pred   = P_t*ones(N,1);

    % Resolver MPC
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; ETc_pred; P_pred] ...
    );

    w_opt = full(sol.x);
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);

    % Evolución del sistema
    xk = full(f(xk,uk,ETc_t,P_t));

    % Guardar
    theta_hist(t) = xk;
    u_hist(t)     = uk;
    ETc_hist(t)   = ETc_t;
    P_hist(t)     = P_t;

    % Horizonte recedente
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Gráficas
% -------------------------
time = 1:Tsim;
figure('Color','w') % Fondo de ventana blanco

% --- Subplot 1 ---
subplot(3,1,1)
plot(time,theta_hist,'b','LineWidth',2); hold on;
yline(theta_ref,'r--','Referencia');
ylabel('\theta')
title('Humedad del suelo')
grid on
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k', 'GridAlpha', 0.2)

% --- Subplot 2 ---
subplot(3,1,2)
stairs(time,u_hist,'k','LineWidth',2)
ylabel('Riego (mm/día)')
title('Acción de control MPC')
grid on
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k', 'GridAlpha', 0.2)

% --- Subplot 3 ---
subplot(3,1,3)
plot(time,ETc_hist,'LineWidth',2); hold on;
stairs(time,P_hist,'LineWidth',2)
ylabel('mm/día')
xlabel('Tiempo (días)')
legend('ETc(t)','Precipitación')
title('Perturbaciones climáticas')
grid on
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', 'GridColor', 'k', 'GridAlpha', 0.2)