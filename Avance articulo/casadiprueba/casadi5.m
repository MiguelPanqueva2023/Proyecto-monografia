%% MPC para balance hídrico con D(theta)

addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*


%Se usa un factor 1/1000 para mantener coherencia de unidades.
%Luego de corregir la escala del modelo, la humedad del suelo permanece en rangos físicamente válidos. 
% El MPC responde adecuadamente a la evapotranspiración variable y a eventos de lluvia, ajustando el riego para compensar las pérdidas. 
% La referencia no se alcanza completamente debido a restricciones operativas y condiciones climáticas, lo cual refleja un comportamiento realista del sistema.
%¿Por qué θ no llega a la referencia 0.25?
%Porque no es físicamente alcanzable con:
%ETc elevada
%Riego limitado
%Drenaje activo
%El MPC no hace milagros, respeta:
%Límites de riego
%Dinámica del suelo
%Penalización energética
%Esto no es un error, es realismo del modelo.

%% -------------------------
% Parámetros del suelo
% -------------------------
Zr = 0.30;        % Profundidad radicular [m]
dt = 1.0;         % Paso de tiempo [día]

mm_to_m = 1e-3; 

theta_ref = 0.25; % Humedad objetivo
theta_fc  = 0.26; % Capacidad de campo
kd        = 2.0;   % Coeficiente de drenaje

%% -------------------------
% Horizonte MPC
% -------------------------
N = 10;

%% -------------------------
% Definición simbólica
% -------------------------
x  = SX.sym('x');      % Humedad del suelo
u  = SX.sym('u');      % Riego
P  = SX.sym('P');      % Precipitación
ET = SX.sym('ET');     % ETc(t)

%% -------------------------
% Drenaje no lineal D(theta)
% -------------------------
D_theta = if_else(x > theta_fc, kd*(x - theta_fc), 0);

%% -------------------------
% Modelo dinámico
% -------------------------
x_next = x + dt*(mm_to_m/Zr)*(u + P - ET - D_theta);
f = Function('f',{x,u,P,ET},{x_next});

%% -------------------------
% Variables de optimización
% -------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);

X0     = SX.sym('X0');        % Estado inicial
P_h    = SX.sym('P_h',N);     % Precipitación futura
ETc_h  = SX.sym('ETc_h',N);   % ETc futura

%% -------------------------
% Función objetivo y restricciones
% -------------------------
J = 0;
g = [];

% Condición inicial
g = [g; X(1) - X0];

for k = 1:N
    J = J + (X(k)-theta_ref)^2 + 0.01*U(k)^2;
    g = [g; X(k+1) - f(X(k),U(k),P_h(k),ETc_h(k))];
end

%% -------------------------
% Formulación NLP
% -------------------------
OPT = struct('x',[X;U],...
             'f',J,...
             'g',g,...
             'p',[X0; P_h; ETc_h]);

opts.ipopt.print_level = 0;
opts.print_time = false;

solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Límites
% -------------------------
lbg = zeros(N+1,1);
ubg = zeros(N+1,1);

lbw = [ ...
    0.15*ones(N+1,1);  % theta mínima
    0.0*ones(N,1)      % riego mínimo
];

ubw = [ ...
    0.35*ones(N+1,1);  % theta máxima
    6.0*ones(N,1)     % riego máximo
];

%% -------------------------
% Simulación en lazo cerrado
% -------------------------
Tsim = 60;
xk = 0.18;

theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);
ETc_hist   = zeros(Tsim,1);
P_hist     = zeros(Tsim,1);

w0 = zeros((N+1)+N,1);

for t = 1:Tsim

    % ETc(t) variable
    ETc_t = 4 + sin(2*pi*t/30);
    ETc_hist(t) = ETc_t;

    % Precipitación por eventos
    P_t = 0;
    if (t>=10 && t<=12) || (t>=20 && t<=22)
        P_t = 4;
    end
    P_hist(t) = P_t;

    % Horizontes
    P_h_val   = P_t*ones(N,1);
    ETc_h_val = ETc_t*ones(N,1);

    % Resolver MPC
    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; P_h_val; ETc_h_val] ...
    );

    w_opt = full(sol.x);
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);

    % Aplicar dinámica
    xk = full(f(xk,uk,P_t,ETc_t));

    theta_hist(t) = xk;
    u_hist(t)     = uk;

    % Horizonte recedente
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Gráficas
% -------------------------
time = 1:Tsim;
fs = 18;          % Tamaño de letra maestro
fs_tit = fs + 4;  % Tamaño para títulos
negro = [0, 0, 0];
azulMedio = [0, 0.4, 0.8];
azulStd   = [0, 0.447, 0.741];
rojoStd   = [0.85, 0.325, 0.098];

% Crear figura con tamaño optimizado para 60 días
figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Humedad del suelo ---
subplot(3,1,1)
plot(time, theta_hist, 'Color', azulMedio, 'LineWidth', 3); hold on;

% Referencia
yline(theta_ref, 'r--', 'Referencia', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Restricciones de Humedad (0.15 - 0.35)
yline(0.15, 'k:', 'Mín. (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Máx. (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

ylim([0.10, 0.40]); % Margen para legibilidad
ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del Suelo (Drenaje No Lineal)', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Acción de control MPC ---
subplot(3,1,2)
stairs(time, u_hist, 'Color', negro, 'LineWidth', 3); hold on;

% Restricción de Riego Máximo (10 mm/día)
yline(6.0, 'r:', 'Límite Riego (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); % Espacio extra para la etiqueta del límite
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de Control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Perturbaciones climáticas ---
subplot(3,1,3)
plot(time, ETc_hist, 'Color', azulStd, 'LineWidth', 3); hold on;
stairs(time, P_hist, 'Color', rojoStd, 'LineWidth', 3);

% Leyenda (Grande, Fondo Blanco, Letra Negra)
lgd = legend('ETc(t)', 'Precipitación');
lgd.FontSize = fs;
lgd.TextColor = negro;
lgd.Color = 'w';
lgd.EdgeColor = negro;

ylim([-0.5, max(ETc_hist) + 4]); 
ylabel('mm/día', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Perturbaciones Climáticas', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);


