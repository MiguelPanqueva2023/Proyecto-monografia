%% MPC para balance hídrico del suelo
% ETc dependiente del tiempo (prueba inicial)
% ------------------------------------------------
% Esta versión introduce:
%  - Precipitación como perturbación externa
%  - Evapotranspiración ETc variable en el tiempo
%
% El modelo sigue siendo simple y explicable
% Autor: Miguel Panqueva (ejemplo académico)

addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;

import casadi.*

%ETc se vuelve dependiente de t (variable) se usa una funcion sinusoidal
%para esto (comportamiento suave), debe influir en el riego, con ETc alta
%mas riego, con ETc baja menos riego

%% -------------------------
% Parámetros físicos del sistema
% --------------------------
Zr = 0.30;      % Profundidad radicular [m]
dt = 1.0;       % Paso de tiempo [día]

D = 0.5;        % Drenaje constante [mm/día]
theta_ref = 0.25; % Humedad objetivo

%% -------------------------
% Horizonte MPC
% --------------------------
N = 10;         % Horizonte de predicción

%% -------------------------
% Definición simbólica
% --------------------------
x = SX.sym('x');     % Estado: humedad del suelo
u = SX.sym('u');     % Control: riego

p  = SX.sym('p');    % Precipitación
etc = SX.sym('etc'); % Evapotranspiración (dependiente del tiempo)

% Modelo dinámico discreto (Euler)
% Balance hídrico simplificado
x_next = x + dt*(1/Zr)*(u + p - etc - D);
f = Function('f',{x,u,p,etc},{x_next});

%% -------------------------
% Variables de optimización
% --------------------------
X = SX.sym('X',N+1);   % Estados en el horizonte
U = SX.sym('U',N);     % Controles
P = SX.sym('P',N);     % Precipitación en el horizonte
ETC = SX.sym('ETC',N); % ETc en el horizonte

X0 = SX.sym('X0');     % Estado inicial

%% -------------------------
% Función objetivo y restricciones
% --------------------------
J = 0;
g = [];

% Condición inicial
g = [g; X(1) - X0];

for k = 1:N
    % Función de costo:
    % - Mantener theta cerca de la referencia
    % - Penalizar riego excesivo
    J = J + (X(k)-theta_ref)^2 + 0.01*U(k)^2;

    % Dinámica del sistema
    g = [g; X(k+1) - f(X(k),U(k),P(k),ETC(k))];
end

%% -------------------------
% Formulación del NLP
% --------------------------
OPT = struct( ...
    'x',[X;U], ...
    'f',J, ...
    'g',g, ...
    'p',[X0; P; ETC] ...
);

opts.ipopt.print_level = 0;
opts.print_time = false;
solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Límites
% --------------------------
lbg = zeros(N+1,1);
ubg = zeros(N+1,1);

lbw = [ ...
    0.15*ones(N+1,1);   % Humedad mínima
    0.0*ones(N,1)       % Riego mínimo
];

ubw = [ ...
    0.35*ones(N+1,1);   % Humedad máxima
    6.0*ones(N,1)      % Riego máximo
];

%% -------------------------
% Simulación en lazo cerrado
% --------------------------
Tsim = 30;          % Días de simulación
xk = 0.18;          % Humedad inicial

theta_hist = zeros(Tsim,1);
u_hist = zeros(Tsim,1);

% Inicialización del solver
w0 = zeros((N+1)+N,1);

%% -------------------------
% Definición de perturbaciones
% --------------------------

% Precipitación simulada (lluvias intermitentes)
Psim = zeros(Tsim,1);
Psim(10:12) = 4;
Psim(20:22) = 3;

% Evapotranspiración variable en el tiempo
% Variación suave tipo climática
t = (1:Tsim)';
ETc_sim = 4 + 1.5*sin(2*pi*t/30);

%% -------------------------
% Bucle de simulación MPC
% --------------------------
for k = 1:Tsim

    % Extraer horizonte de perturbaciones
    P_h = Psim(k:min(k+N-1,Tsim));
    ETc_h = ETc_sim(k:min(k+N-1,Tsim));

    % Relleno si se alcanza el final
    if length(P_h) < N
        P_h(end+1:N) = P_h(end);
        ETc_h(end+1:N) = ETc_h(end);
    end

    % Resolver MPC
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; P_h(:); ETc_h(:)] ...
    );

    w_opt = full(sol.x);
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);

    % Aplicar control
    xk = full(f(xk,uk,Psim(k),ETc_sim(k)));

    % Guardar resultados
    theta_hist(k) = xk;
    u_hist(k) = uk;

    % Horizonte recedente
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Gráficas
% --------------------------
time = 1:Tsim;
fs = 18;          % Tamaño de letra maestro (Grande)
fs_tit = fs + 4;  % Tamaño para títulos (Extra Grande)
negro = [0, 0, 0];
azulMedio = [0, 0.4, 0.8];
azulStd   = [0, 0.447, 0.741];
rojoStd   = [0.85, 0.325, 0.098];

% Crear figura con tamaño amplio
figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Humedad del suelo ---
subplot(3,1,1)
plot(time, theta_hist, 'Color', azulMedio, 'LineWidth', 3); hold on;

% Línea de Referencia
yline(theta_ref, 'r--', 'Referencia', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Restricciones de Humedad (Basadas en lbw/ubw)
yline(0.15, 'k:', 'Mín. (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Máx. (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0.10, 0.40]); % Margen para evitar cortes
ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del suelo', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Acción de control MPC ---
subplot(3,1,2)
stairs(time, u_hist, 'Color', negro, 'LineWidth', 3); hold on;

% Restricción de Riego Máximo (10 mm/día)
yline(6.0, 'r:', 'Límite Riego (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); % Margen superior para el límite
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de Control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Perturbaciones climáticas ---
subplot(3,1,3)
plot(time, ETc_sim, 'Color', azulStd, 'LineWidth', 3); hold on;
stairs(time, Psim, 'Color', rojoStd, 'LineWidth', 3);

% Configuración de Leyenda (Grande, Fondo Blanco, Letra Negra)
lgd = legend('ETc(t)', 'Precipitación');
lgd.FontSize = fs;
lgd.TextColor = negro;
lgd.Color = 'w';
lgd.EdgeColor = negro;

ylim([-0.5, max(ETc_sim) + 4]); % Margen dinámico
ylabel('mm/día', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Perturbaciones climáticas', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);