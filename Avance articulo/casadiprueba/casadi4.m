%% NMPC para balance hídrico del suelo
% ETc dependiente de la humedad del suelo (theta)
% Implementación inicial - validación conceptual

addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

%ETc se vuelve dependiente de theta (la humedad del suelo), se refactoriza
%el termino correspondiente en la ecuacion original, Etc,pot(t) y K(θ)
%Ks(θ) es una funcion auxiliar  que captura el estres hidrico, no es
%una nueva variable, se utiliza para refactorizar ETc(theta,t), ETc,pot(t) representa la demanda atmosférica y  
% Ks(θ) modela la reducción de dicha demanda en función del contenido de humedad del suelo
% Esta reformulación permite introducir no linealidad en el modelo de forma controlada y 
% facilita su integración dentro del esquema de Control Predictivo No Lineal (NMPC).

%% -------------------------
% Parámetros físicos
% -------------------------
Zr = 0.30;        % Profundidad radicular [m]
dt = 1.0;         % Paso de tiempo [día]
D  = 0.5;         % Drenaje profundo [mm/día]

theta_fc = 0.30; % Capacidad de campo
theta_wp = 0.15; % Punto de marchitez

theta_ref = 0.25; % Humedad objetivo

%% -------------------------
% Horizonte MPC
% -------------------------
N = 10;

%% -------------------------
% Variables simbólicas
% -------------------------
x = SX.sym('x');      % Humedad del suelo
u = SX.sym('u');      % Riego
ETc = SX.sym('ETc');  % Evapotranspiración potencial
P   = SX.sym('P');    % Precipitación

%% -------------------------
% Factor de estrés hídrico Ks(theta)
% -------------------------
Ks = if_else( ...
    x >= theta_fc, 1, ...
    if_else(x <= theta_wp, 0, ...
    (x - theta_wp)/(theta_fc - theta_wp)));

%% -------------------------
% Modelo dinámico (Euler)
% -------------------------
x_next = x + dt*(1/Zr)*(1/1000)*(u + P - ETc - D);
f = Function('f',{x,u,ETc,P},{x_next});

%% -------------------------
% Variables de optimización
% -------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);

% Parámetros: estado inicial + perturbaciones
X0 = SX.sym('X0');
ETc_h = SX.sym('ETc_h',N);
P_h   = SX.sym('P_h',N);

p = [X0; ETc_h; P_h];

%% -------------------------
% Función objetivo y restricciones
% -------------------------
J = 0;
g = [];

g = [g; X(1) - X0];

for k = 1:N
    J = J + (X(k)-theta_ref)^2 + 0.01*U(k)^2;
    g = [g; X(k+1) - f(X(k),U(k),ETc_h(k),P_h(k))];
end

%% -------------------------
% Formulación NLP
% -------------------------
OPT = struct('x',[X;U],'f',J,'g',g,'p',p);

opts.ipopt.print_level = 0;
opts.print_time = false;
solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Límites
% -------------------------
lbg = zeros(N+1,1);
ubg = zeros(N+1,1);

lbw = [0.15*ones(N+1,1); 0*ones(N,1)];
ubw = [0.35*ones(N+1,1); 6*ones(N,1)];

%% -------------------------
% Simulación en lazo cerrado
% -------------------------
Tsim = 30;
xk = 0.18;

theta_hist = zeros(Tsim,1);
u_hist = zeros(Tsim,1);

w0 = zeros((N+1)+N,1);

for t = 1:Tsim

    % ETc variable (demanda climática)
    ETc_val = 4 + sin(2*pi*t/30);

    % Evento de lluvia
    if (t >= 10 && t <= 12) || (t >= 20 && t <= 22)
        P_val = 4;
    else
        P_val = 0;
    end

    ETc_vec = ETc_val*ones(N,1);
    P_vec   = P_val*ones(N,1);

    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; ETc_vec; P_vec] ...
    );

    w_opt = full(sol.x);
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);

    xk = full(f(xk,uk,ETc_val,P_val));

    theta_hist(t) = xk;
    u_hist(t) = uk;

    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Gráficas
% -------------------------
time = 1:Tsim;
fs = 18;          % Tamaño de letra maestro (Grande y legible)
fs_tit = fs + 4;  % Tamaño para títulos
negro = [0, 0, 0];
azulMedio = [0, 0.4, 0.8];
azulStd   = [0, 0.447, 0.741];
rojoStd   = [0.85, 0.325, 0.098];

% Crear figura amplia
figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Humedad del suelo (Estado) ---
subplot(3,1,1)
plot(time, theta_hist, 'Color', azulMedio, 'LineWidth', 3); hold on;

% Línea de Referencia
yline(theta_ref, 'r--', 'Referencia', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Restricciones de Humedad (lbw/ubw)
yline(0.15, 'k:', 'Mín. (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Máx. (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0.10, 0.40]); % Margen para legibilidad
ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del Suelo (NMPC)', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Acción de control (Riego) ---
subplot(3,1,2)
stairs(time, u_hist, 'Color', negro, 'LineWidth', 3); hold on;

% Restricción de Riego Máximo
yline(6.0, 'r:', 'Límite Riego (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); % Espacio extra arriba
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de Control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Perturbaciones (Demanda y Lluvia) ---
subplot(3,1,3)
% Re-calculamos para el plot consistente
ETc_plot = 4 + sin(2*pi*time/30);
P_plot = zeros(size(time));
P_plot((time >= 10 & time <= 12) | (time >= 20 & time <= 22)) = 4;

plot(time, ETc_plot, 'Color', azulStd, 'LineWidth', 3); hold on;
stairs(time, P_plot, 'Color', rojoStd, 'LineWidth', 3);

% Leyenda con formato corregido
lgd = legend('ETc(t)', 'Precipitación');
lgd.FontSize = fs;
lgd.TextColor = negro;
lgd.Color = 'w';
lgd.EdgeColor = negro;

ylim([-0.5, max(ETc_plot) + 4]); 
ylabel('mm/día', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Perturbaciones Climáticas', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);
