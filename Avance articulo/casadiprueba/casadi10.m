%% MPC para balance hídrico del suelo
% Modelo conceptual con saturaciones suaves
% Objetivo: validación estructural del MPC (sin grey-box aún)


addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');


clear; clc; close all;
import casadi.*

%La precipitación no se modeló como un escalón ideal, sino como una función sigmoidal doble que representa el frente de entrada, el núcleo y la disipación de una tormenta. 
% Esto permite una excitación climática físicamente realista y diferenciable, adecuada para control predictivo. 
% La precipitación fue modelada mediante una función sigmoide doble que representa la formación, desarrollo y disipación de un evento de lluvia. 
% Esto evita discontinuidades artificiales y permite que el MPC explote su capacidad predictiva al anticipar la llegada de agua al sistema suelo–planta.


%% -------------------------
% Parámetros físicos del suelo
% --------------------------
Zr = 0.30;        % Profundidad radicular [m]
dt = 1.0;         % Paso de tiempo [día]

theta_wp  = 0.12; % Punto de marchitez
theta_fc  = 0.25; % Capacidad de campo
theta_sat = 0.35; % Saturación

theta_ref = 0.25; % Referencia deseada

%% -------------------------
% Horizonte MPC
% --------------------------
N = 10;

%% -------------------------
% Pesos del costo
% --------------------------
w_theta = 10;     % Seguimiento
w_u     = 0.01;   % Energía de riego
w_du    = 0.5;    % Suavidad del riego
w_int   = 5;      % Acción integral

w_low   = 50;     % Saturación inferior
w_high  = 50;     % Saturación superior

alpha_sat = 50;   % Suavidad de saturaciones

%% -------------------------
% Funciones suaves
% --------------------------
softplus = @(x) (1/alpha_sat)*log(1 + exp(alpha_sat*x));

%% -------------------------
% Modelo simbólico
% --------------------------
x  = SX.sym('x');    % Humedad
u  = SX.sym('u');    % Riego
p  = SX.sym('p',2);  % [Precipitación; ETc]

P_rain = p(1);
ETc    = p(2);

% ETc dependiente de theta (estrés hídrico simple)
Ks = fmin(1, fmax(0, (x - theta_wp)/(theta_fc - theta_wp)));
ETc_eff = Ks * ETc;

% Drenaje dependiente de theta (suave)
K_d = 10;
D_theta = softplus(x - theta_fc) * K_d;

% Modelo dinámico (Euler)
x_next = x + dt*(1/(Zr*1000))*(u + P_rain - ETc_eff - D_theta);
f = Function('f',{x,u,p},{x_next});

%% -------------------------
% Variables de optimización
% --------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);
I = SX.sym('I',N+1); % estado integral

X0 = SX.sym('X0');
I0 = SX.sym('I0');
P_h = SX.sym('P_h',N);
ETc_h = SX.sym('ETc_h',N);

%% -------------------------
% Función objetivo y restricciones
% --------------------------
J = 0;
g = [];

% Condiciones iniciales
g = [g;
     X(1) - X0;
     I(1) - I0];

for k = 1:N

    % Error
    e = X(k) - theta_ref;

    % Costo principal
    J = J + w_theta*e^2 + w_u*U(k)^2 + w_int*I(k)^2;

    % Saturaciones suaves de humedad
    J = J + w_low  * softplus(theta_wp - X(k))^2;
    J = J + w_high * softplus(X(k) - theta_fc)^2;

    % Suavidad del riego
    if k > 1
        J = J + w_du*(U(k) - U(k-1))^2;
    end

    % Dinámica
    g = [g;
         X(k+1) - f(X(k), U(k), [P_h(k); ETc_h(k)]);
         I(k+1) - (I(k) + e)];
end

%% -------------------------
% Formulación NLP
% --------------------------
OPT_variables = [X; U; I];
OPT = struct('x',OPT_variables, ...
             'f',J, ...
             'g',g, ...
             'p',[X0; I0; P_h; ETc_h]);

opts.ipopt.print_level = 0;
opts.print_time = false;
solver = nlpsol('solver','ipopt',OPT,opts);

%% -------------------------
% Límites
% --------------------------
lbg = zeros(size(g));
ubg = zeros(size(g));

lbw = [ ...
    0.05*ones(N+1,1);      % theta
    0.0*ones(N,1);         % riego
    -inf*ones(N+1,1) ];    % integral

ubw = [ ...
    0.40*ones(N+1,1);
    6.0*ones(N,1);
    inf*ones(N+1,1) ];

%% -------------------------
% Simulación en lazo cerrado
% --------------------------
Tsim = 60;
xk = 0.18;
ik = 0;

theta_hist = zeros(Tsim,1);
u_hist     = zeros(Tsim,1);
ETc_hist   = zeros(Tsim,1);
P_hist     = zeros(Tsim,1);

w0 = zeros(length(OPT_variables),1);

for t = 1:Tsim

    % Perturbaciones
    ETc_t = 4 + sin(2*pi*t/30);
    % Lluvia física (doble sigmoide)
    % -------------------------
    Pmax = 4;       % mm/día (intensidad máxima)
    t1   = 10;      % inicio de evento
    t2   = 13;      % fin de evento
    tau  = 0.5;     % suavidad del frente (días)
    
    sig = @(x) 1./(1 + exp(-x));
    
    P_event1 = Pmax * ( sig((t - t1)/tau) - sig((t - t2)/tau) );
    
    % Segundo evento de lluvia
    t3 = 20;
    t4 = 23;
    P_event2 = Pmax * ( sig((t - t3)/tau) - sig((t - t4)/tau) );
    
    P_t = P_event1 + P_event2;

    sol = solver( ...
        'x0', w0, ...
        'lbx', lbw, ...
        'ubx', ubw, ...
        'lbg', lbg, ...
        'ubg', ubg, ...
        'p', [xk; ik; P_t*ones(N,1); ETc_t*ones(N,1)] );

    w_opt = full(sol.x);
    U_opt = w_opt(N+2:N+1+N);

    uk = U_opt(1);

    % Actualizar sistema
    xk = full(f(xk, uk, [P_t; ETc_t]));
    ik = ik + (xk - theta_ref);

    % Guardar
    theta_hist(t) = xk;
    u_hist(t)     = uk;
    ETc_hist(t)   = ETc_t;
    P_hist(t)     = P_t;

    % Horizonte recedente
    w0 = w_opt;
end

%% -------------------------
% Gráficas
% --------------------------
time = 1:Tsim;
fs = 18;          % Tamaño de letra grande
fs_tit = fs + 4;  % Títulos más grandes
negro = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Humedad del suelo ---
subplot(3,1,1)
plot(time, theta_hist, 'b', 'LineWidth', 3); hold on;
% Usamos theta_ref porque está definida en tu línea 24
yline(theta_ref, 'r--', 'Referencia', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

% Ajuste de límites para que la referencia 0.25 no quede pegada arriba
ylim([0.10, 0.40]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del suelo', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 2: Acción de control ---
subplot(3,1,2)
stairs(time, u_hist, 'k', 'LineWidth', 3); hold on;
% El límite de riego en este script es 6.0 (definido en lbw/ubw)
yline(6.0, 'r:', 'Límite (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold');

ylim([-0.5, 8]); % Margen para que el límite 6 se vea bien
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);

% --- Subplot 3: Perturbaciones ---
subplot(3,1,3)
plot(time, ETc_hist, 'LineWidth', 3); hold on;
plot(time, P_hist, 'LineWidth', 3);

ylim([-0.5, max([max(ETc_hist), max(P_hist)]) + 3]);
ylabel('mm/día', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Perturbaciones climáticas', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');

lgd = legend('ETc(t)', 'Precipitación');
lgd.FontSize = fs;
lgd.TextColor = negro;
lgd.Color = 'w';
lgd.EdgeColor = negro;
grid on;
set(gca, 'FontSize', fs, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridAlpha', 0.2, 'LineWidth', 1.5);