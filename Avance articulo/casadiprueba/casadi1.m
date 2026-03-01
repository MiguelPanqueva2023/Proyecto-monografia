%% MPC para balance hídrico del suelo (ejemplo inicial)
% Modelo simple - valores simulados
% Objetivo: verificar comportamiento inicial del MPC
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');   % ajusta a la carpeta donde descomprimiste CasADi
clear; clc; close all;
import casadi.*

%%Prueba con ETc y D constantes, no dependen ni  de theta ni de t,
%%precipitacion eliminada para verificar funcionamiento en entorno
%%controlado sin lluvia, solo depende del control de riego, solo se maneja
%%theta como variable de estado, t aun no


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
x = SX.sym('x');  % Estado: humedad del suelo
u = SX.sym('u');  % Control: riego [mm/día]

% Modelo dinámico (Euler)
x_next = x + dt*(1/Zr)*((u - ETc - D)/1000);
f = Function('f',{x,u},{x_next});

%% -------------------------
% Variables de optimización
% --------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);

X0 = SX.sym('X0');   % Estado inicial (parámetro)

%% -------------------------
% Función objetivo y restricciones
% --------------------------
Q = 100;      % Peso del error de humedad
R = 0.001;    % Peso del riego

J = 0;
g = [];

% Condición inicial
g = [g; X(1) - X0];

for k = 1:N
    % Costo
    J = J + Q*(X(k) - theta_ref)^2 + R*U(k)^2;

    % Dinámica
    g = [g; X(k+1) - f(X(k),U(k))];
end

%% -------------------------
% Formulación NLP
% --------------------------
OPT = struct('x',[X;U],'f',J,'g',g,'p',X0);

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
Tsim = 30;             % días de simulación
xk = 0.18;             % humedad inicial

theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);

% Inicialización del horizonte (IMPORTANTE)
w0 = [ ...
    xk*ones(N+1,1);
    zeros(N,1)
];

for t = 1:Tsim
    % Resolver MPC
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', xk ...
    );

    w_opt = full(sol.x);

    % Extraer control óptimo
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);

    % Aplicar control a la planta
    xk = full(f(xk,uk));

    % Guardar resultados
    theta_hist(t) = xk;
    u_hist(t) = uk;

    % Horizonte recedente
    w0 = [ ...
        w_opt(2:N+1);
        w_opt(N+1);
        U_opt(2:end);
        U_opt(end)
    ];
end

%% -------------------------
% Gráficas
% --------------------------
time = 1:Tsim;
fs = 18;          % <--- TAMAÑO MAESTRO: Aumentado a 18 para máxima visibilidad
fs_tit = fs + 4;  % Tamaño para títulos
negro = [0, 0, 0];
azulMedio = [0, 0.4, 0.8];

figure('Color','w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]); % Ventana más grande

% --- Subplot 1: Humedad del suelo ---
subplot(2,1,1)
plot(time, theta_hist, 'Color', azulMedio, 'LineWidth', 3); hold on;

% Línea de Referencia con texto grande
yline(theta_ref, 'r--', 'Referencia', 'LineWidth', 2, ...
    'FontSize', fs, 'FontWeight', 'bold', 'LabelVerticalAlignment', 'bottom');

% Límites de Restricción con etiquetas grandes
yline(0.15, 'k:', 'Mínimo (0.15)', 'Alpha', 0.6, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Máximo (0.35)', 'Alpha', 0.6, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0.10, 0.40]); 
ylabel('$\theta$', 'Interpreter', 'latex', 'FontSize', fs+2, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del suelo', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;

% Aplicar tamaño a los números de los ejes (Ticks)
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Acción de control ---
subplot(2,1,2)
stairs(time, u_hist, 'Color', negro, 'LineWidth', 3); hold on;

% Límite de Riego con texto grande
yline(6.0, 'r:', 'Límite Riego (6)', 'LineWidth', 2, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]);
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de Control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;

% Aplicar tamaño a los números de los ejes (Ticks)
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);