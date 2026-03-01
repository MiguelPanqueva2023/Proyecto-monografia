%% MPC para balance hídrico del suelo con precipitación
% Modelo inicial con perturbación externa
% Objetivo: evaluar reacción del MPC ante lluvia
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

%se añade la precipitacion como perturbacion externa, ya no solo depende
%del control del riego, con lluvia el riego baja, se logra estabilidad en
%valores menores, la lluvia 'ayuda', sigue sin depender de t y de theta

%% -------------------------
% Parámetros del sistema
% --------------------------
Zr = 0.30;        % Profundidad radicular [m]
dt = 1.0;         % Paso de tiempo [día]

ETc = 4.0;        % Evapotranspiración [mm/día]
D   = 0.5;        % Drenaje [mm/día]

theta_ref = 0.25; % Humedad deseada

%% -------------------------
% Horizonte MPC
% --------------------------
N = 10;           % Horizonte de predicción

%% -------------------------
% Definición simbólica
% --------------------------
x = SX.sym('x');        % Estado
u = SX.sym('u');        % Control
p = SX.sym('p');        % Precipitación (parámetro)

% Modelo dinámico
x_next = x + dt*(1/(Zr*1000))*(u + p - ETc - D);
f = Function('f',{x,u,p},{x_next});

%% -------------------------
% Variables de optimización
% --------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);

X0 = SX.sym('X0');         % Estado inicial
P  = SX.sym('P',N);        % Precipitación en el horizonte

%% -------------------------
% Función objetivo y restricciones
% --------------------------
J = 0;
g = [];

% Condición inicial
g = [g; X(1) - X0];

for k = 1:N
    % Función de costo
    J = J + (X(k)-theta_ref)^2 + 0.01*U(k)^2;

    % Dinámica con precipitación
    g = [g; X(k+1) - f(X(k),U(k),P(k))];
end

%% -------------------------
% Formulación NLP
% --------------------------
OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0; P]);

opts.ipopt.print_level = 0;
opts.print_time = false;

solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Límites
% --------------------------
lbg = zeros(N+1,1);
ubg = zeros(N+1,1);

lbw = [ ...
    0.15*ones(N+1,1);   % theta mínima
    0.0*ones(N,1)       % riego mínimo
];

ubw = [ ...
    0.35*ones(N+1,1);   % theta máxima
    6.0*ones(N,1)      % riego máximo
];

%% -------------------------
% Simulación en lazo cerrado
% --------------------------
Tsim = 30;             
xk = 0.18;             % humedad inicial

theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);
p_hist     = zeros(Tsim,1);

% Perfil de lluvia (ejemplo)
P_sim = zeros(Tsim,1);
P_sim(10:12) = 6;      % Evento de lluvia [mm/día]

% Inicialización
w0 = zeros((N+1)+N,1);

for t = 1:Tsim

    % Precipitación en el horizonte
    if t+N-1 <= Tsim
        P_hor = P_sim(t:t+N-1);
    else
        P_hor = [P_sim(t:end); zeros(t+N-1-Tsim,1)];
    end

    % Resolver MPC
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; P_hor] ...
    );

    w_opt = full(sol.x);

    % Control óptimo
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);

    % Aplicar dinámica real
    xk = full(f(xk,uk,P_sim(t)));

    % Guardar
    theta_hist(t) = xk;
    u_hist(t) = uk;
    p_hist(t) = P_sim(t);

    % Horizonte recedente
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Gráficas
% --------------------------
time = 1:Tsim;
fs = 18;          % Tamaño maestro para máxima legibilidad
fs_tit = fs + 4;  % Tamaño para títulos
negro = [0, 0, 0];
azulMedio = [0, 0.4, 0.8];
rojoStd   = [0.85, 0.325, 0.098]; % Color para la lluvia

% Crear figura grande
figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Humedad del suelo ---
subplot(3,1,1)
plot(time, theta_hist, 'Color', azulMedio, 'LineWidth', 3); hold on;

% Referencia
yline(theta_ref, 'r--', 'Referencia', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Restricciones de Humedad
yline(0.15, 'k:', 'Mínimo (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Máximo (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0.10, 0.40]); % Margen para que se vean los límites
ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del suelo', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Acción de control MPC ---
subplot(3,1,2)
stairs(time, u_hist, 'Color', negro, 'LineWidth', 3); hold on;

% Restricción de Riego
yline(6.0, 'r:', 'Límite Riego (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); % Margen para que el 10 no toque el techo
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de Control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Perturbación externa (Lluvia) ---
subplot(3,1,3)
stairs(time, p_hist, 'Color', rojoStd, 'LineWidth', 3);
ylim([-0.5, max(p_hist) + 4]); % Margen dinámico según la lluvia caída
ylabel('Lluvia (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Perturbación externa (Lluvia)', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);
