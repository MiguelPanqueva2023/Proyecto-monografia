clc; clear; close all
import casadi.*

%% ============================================================
% 1. CARGAR RED ENTRENADA (copiada de tu script)
% ============================================================
load NN_residual.mat   % (si quieres, luego lo guardamos automático)

% Contiene: W1,b1,W2,b2,W3,b3,xmin,xmax,ymin,ymax

%% ============================================================
% 2. MODELO FÍSICO
% ============================================================
Zr = 0.30;
dt = 1;

theta_wp = 0.12;
theta_fc = 0.25;

x = SX.sym('x');
u = SX.sym('u');
p = SX.sym('p',2);   % [Rain, ETc]

P = p(1);
ETc = p(2);

Ks = fmin(1,fmax(0,(x-theta_wp)/(theta_fc-theta_wp)));
ETc_eff = Ks*ETc;

K_d = 10;
D = log(1+exp(10*(x-theta_fc)));

x_next = x + dt*(1/(Zr*1000))*(u + P - ETc_eff - D);
f = Function('f',{x,u,p},{x_next});

%% ============================================================
% 3. MPC
% ============================================================
N = 10;
theta_ref = 0.25;

X = SX.sym('X',N+1);
U = SX.sym('U',N);

X0 = SX.sym('X0');
P_h = SX.sym('P_h',N);
ETc_h = SX.sym('ETc_h',N);

J = 0;
g = [X(1)-X0];

for k=1:N
    J = J + 10*(X(k)-theta_ref)^2 + 0.01*U(k)^2;
    g = [g;
         X(k+1) - f(X(k),U(k),[P_h(k);ETc_h(k)])];
end

OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0;P_h;ETc_h]);
solver = nlpsol('solver','ipopt',OPT);

lbg = zeros(size(g)); ubg=lbg;

lbw = [0.1*ones(N+1,1); 0*ones(N,1)];
ubw = [0.4*ones(N+1,1); 6*ones(N,1)];

%% ============================================================
% 4. SIMULACIÓN GREY-BOX
% ============================================================
T = 40;
xk = 0.18;

theta_hist = zeros(T,1);
theta_phys = zeros(T,1);
u_hist = zeros(T,1);

w0 = zeros(N+1+N,1);

for t=1:T

    P_t = (t>=10 && t<=14)*4;
    ETc_t = 4 + sin(2*pi*t/30);

    P_hor = P_t*ones(N,1);
    ETc_hor = ETc_t*ones(N,1);

    sol = solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,...
                 'p',[xk;P_hor;ETc_hor]);

    w_opt = full(sol.x);
    uk = w_opt(N+2);

    % ---- modelo físico
    theta_f = full(f(xk,uk,[P_t;ETc_t]));

    % ---- red neuronal residual
    x_nn = [xk; uk; P_t; ETc_t];
    x_n = (x_nn - xmin)./(xmax - xmin + eps);

    a1 = tanh(W1*x_n + b1);
    a2 = tanh(W2*a1 + b2);
    r_n = W3*a2 + b3;
    r = r_n*(ymax - ymin) + ymin;

    % ---- grey-box
    xk = theta_f + r;

    theta_hist(t) = xk;
    theta_phys(t) = theta_f;
    u_hist(t) = uk;

    w0 = w_opt;
end

%% ============================================================
% 5. GRÁFICAS
% ============================================================
t_vec = 1:T;
fs = 18;          % Tamaño de letra maestro
fs_tit = fs + 4;  
negro = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Comparativa Modelo vs Grey-Box ---
subplot(2,1,1)
plot(t_vec, theta_phys, '--b', 'LineWidth', 2.5); hold on; % Modelo físico en azul punteado
plot(t_vec, theta_hist, 'r', 'LineWidth', 3);           % Grey-box en rojo sólido
yline(theta_ref, 'k--', 'LineWidth', 2);                % Referencia en negro punteado

% Etiquetas de límites (Siguiendo tu estilo)
yline(theta_wp, 'k:', 'Punto Marchitez', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
yline(0.40, 'k:', 'Saturación', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');

ylim([0.05, 0.50]); 
ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Efecto de la red residual (Grey-Box)', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;

set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

lgd1 = legend('Modelo físico', 'Grey-box', 'Referencia', 'Location', 'northeast');
lgd1.FontSize = fs-2;
lgd1.TextColor = negro;
lgd1.Color = 'w';
lgd1.EdgeColor = negro;

% --- Subplot 2: Acción de Control ---
subplot(2,1,2)
stairs(t_vec, u_hist, 'k', 'LineWidth', 3); hold on;

% Límite de riego (como en tus otros scripts)
yline(6, 'r:', 'Máx. Riego (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold', ...
    'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');

ylim([-0.5, 9]); 
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
grid on;

set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);
