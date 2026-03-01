%% MPC para balance hídrico del suelo
% Modelo físico no lineal suavizado
% ETc y drenaje dependientes de la humedad
% Implementación inicial en CasADi + MATLAB

addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');

clear; clc; close all;
import casadi.*

%un esquema de control predictivo no lineal basado exclusivamente en un modelo físico del balance hídrico del suelo, 
% incorporando no linealidades suavizadas en la evapotranspiración y el drenaje.
%El controlador converge a un equilibrio estable por debajo de la referencia. 
% Este comportamiento es consistente con la formulación actual del problema de optimización, 
% donde el costo asociado al riego domina sobre el error de seguimiento. 
% El resultado confirma la estabilidad del esquema MPC 
% y evidencia la necesidad de un ajuste de pesos y/o restricciones para forzar el cumplimiento de la referencia."

%% -------------------------
% Parámetros físicos
% -------------------------
Zr = 0.30;     % Profundidad radicular [m]
dt = 1.0;      % Paso de tiempo [día]

theta_ref = 0.25;   % Humedad objetivo

%% -------------------------
% Parámetros ETc (suavizado)
% -------------------------
ETc_pot   = 4.0;    % ETc potencial [mm/día]
theta_crit = 0.20;  % Humedad crítica
gamma     = 40;     % Pendiente sigmoide ETc

%% -------------------------
% Parámetros drenaje (suavizado)
% -------------------------
theta_fc = 0.28;    % Capacidad de campo
k_d      = 5.0;     % Coeficiente de drenaje
alpha    = 40;      % Pendiente sigmoide drenaje

%% -------------------------
% Horizonte MPC
% -------------------------
N = 10;

%% -------------------------
% Definición simbólica
% -------------------------
x = SX.sym('x');    % Estado: humedad del suelo
u = SX.sym('u');    % Control: riego

% Factor de estrés hídrico (FAO-56 simplificado)
Ks = 1/(1 + exp(-gamma*(x - theta_crit)));

% Evapotranspiración dependiente de theta
ETc = ETc_pot * Ks;

% Drenaje dependiente de theta
D = k_d*(x - theta_fc)/(1 + exp(-alpha*(x - theta_fc)));

% Modelo dinámico (Euler)
x_next = x + dt*(1/(Zr*1000))*(u - ETc - D);

f = Function('f',{x,u},{x_next});

%% -------------------------
% Variables de optimización
% -------------------------
X = SX.sym('X',N+1);   % Estados
U = SX.sym('U',N);     % Controles
X0 = SX.sym('X0');     % Estado inicial (parámetro)

%% -------------------------
% Función objetivo y restricciones
% -------------------------
J = 0;
g = [];

% Condición inicial
g = [g; X(1) - X0];

for k = 1:N
    % Costo: seguimiento + penalización de riego
    J = J + 10*(X(k)-theta_ref)^2 + 0.001*U(k)^2;


    % Dinámica
    g = [g; X(k+1) - f(X(k),U(k))];
end

%% -------------------------
% Formulación NLP
% -------------------------
OPT = struct('x',[X;U],'f',J,'g',g,'p',X0);

opts.ipopt.print_level = 0;
opts.print_time = false;

solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Límites físicos
% -------------------------
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
% -------------------------
Tsim = 30;      % días
xk = 0.18;      % humedad inicial

theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);

% Inicialización
w0 = zeros((N+1)+N,1);

for t = 1:Tsim

    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', xk ...
    );

    w_opt = full(sol.x);

    % Control óptimo
    U_opt = w_opt(N+2:end);
    uk = U_opt(1);

    % Evolución del sistema
    xk = full(f(xk,uk));

    % Guardar resultados
    theta_hist(t) = xk;
    u_hist(t) = uk;

    % Horizonte recedente
    w0 = [w_opt(2:N+1); w_opt(N+1); U_opt(2:end); U_opt(end)];
end

%% -------------------------
% Gráficas
% -------------------------
time = 1:Tsim;
fs = 18;          % Tamaño de letra maestro (Grande)
fs_tit = fs + 4;  % Tamaño para títulos
negro = [0, 0, 0];
azulMedio = [0, 0.4, 0.8];

% Crear figura amplia
figure('Color','w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

% --- Subplot 1: Humedad del suelo ---
subplot(2,1,1)
plot(time, theta_hist, 'Color', azulMedio, 'LineWidth', 3); hold on;

% Línea de Referencia
yline(theta_ref, 'r--', 'Referencia', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Restricciones de Humedad (0.15 - 0.35)
yline(0.15, 'k:', 'Mín. (0.15)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold'); 
yline(0.35, 'k:', 'Máx. (0.35)', 'Alpha', 0.7, 'FontSize', fs-2, 'FontWeight', 'bold');

% Ajuste de margen para que se vea la brecha con la referencia
ylim([0.10, 0.40]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del Suelo (Modelo No Lineal Suavizado)', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Acción de control MPC ---
subplot(2,1,2)
stairs(time, u_hist, 'Color', negro, 'LineWidth', 3); hold on;

% Restricción de Riego Máximo (10 mm/día)
yline(6.0, 'r:', 'Límite Riego (6)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); % Margen superior para el límite
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de Control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);
