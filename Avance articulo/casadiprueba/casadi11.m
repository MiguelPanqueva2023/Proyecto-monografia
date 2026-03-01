%% MPC hidrológico con referencia dinámica y clima realista
addpath('C:\Users\migue\OneDrive\Documentos\MATLAB\casadiprueba\casadi-3.7.2-windows64-matlab2018b');
clear; clc; close all;
import casadi.*

%% -------------------------
% Parámetros del suelo
% --------------------------
Zr = 0.30;           
dt = 1.0;            

theta_wp  = 0.12;
theta_fc  = 0.25;
theta_sat = 0.40;

%% -------------------------
% Horizonte MPC
% --------------------------
N = 10;

%% -------------------------
% Pesos
% --------------------------
w_theta = 50;
w_u     = 0.01;
w_du    = 0.3;

alpha = 50;

softplus = @(x) (1/alpha)*log(1 + exp(alpha*x));

%% -------------------------
% Variables simbólicas
% --------------------------
x = SX.sym('x');          
u = SX.sym('u');          
p = SX.sym('p',2);        % [P, ETc]

P_rain = p(1);
ETc    = p(2);

%% -------------------------
% Estrés hídrico Ks
% --------------------------
Ks = fmin(1, fmax(0,(x-theta_wp)/(theta_fc-theta_wp)));
ETc_eff = Ks*ETc;

%% -------------------------
% Drenaje suave
% --------------------------
K_d = 10;
D = K_d*softplus(x-theta_fc);

%% -------------------------
% Dinámica
% --------------------------
x_next = x + dt*(1/(Zr*1000))*(u + P_rain - ETc_eff - D);
f = Function('f',{x,u,p},{x_next});

%% -------------------------
% MPC variables
% --------------------------
X = SX.sym('X',N+1);
U = SX.sym('U',N);

X0   = SX.sym('X0');
P_h  = SX.sym('P_h',N);
ETc_h= SX.sym('ETc_h',N);
th_ref = SX.sym('th_ref',N);

%% -------------------------
% Coste
% --------------------------
J = 0;
g = [];
g = [g; X(1)-X0];

for k=1:N
    e = X(k)-th_ref(k);
    J = J + w_theta*e^2 + w_u*U(k)^2;
    if k>1
        J = J + w_du*(U(k)-U(k-1))^2;
    end
    g = [g; X(k+1)-f(X(k),U(k),[P_h(k);ETc_h(k)])];
end

OPT = struct('x',[X;U],'f',J,'g',g,'p',[X0;P_h;ETc_h;th_ref]);

solver = nlpsol('solver','ipopt',OPT,struct('ipopt',struct('print_level',0)));

%% -------------------------
% Límites
% --------------------------
lbg = zeros(size(g)); ubg = lbg;

lbw = [0.10*ones(N+1,1); 0*ones(N,1)];
ubw = [0.40*ones(N+1,1); 6*ones(N,1)];

%% -------------------------
% Clima realista
% --------------------------
Tsim = 60;
xk = 0.18;

ETc_hist = zeros(Tsim,1);
P_hist   = zeros(Tsim,1);
theta_hist = zeros(Tsim,1);
u_hist = zeros(Tsim,1);
theta_ref_hist = zeros(Tsim,1);

t = (1:Tsim)';

ETc_hist = 4 + 1.2*sin(2*pi*t/30);

sig = @(t,t0,w) 1./(1+exp(-10*(t-t0))) - 1./(1+exp(-10*(t-(t0+w))));

P_hist = 3*sig(t,10,4) + 3*sig(t,20,4);

%% -------------------------
% Simulación
% --------------------------
w0 = zeros((N+1)+N,1);

for k=1:Tsim
    
    ETc_k = ETc_hist(k);
    P_k   = P_hist(k);

    % referencia fisiológica
    ETn = min(1,max(0,(ETc_k-2)/(6-2)));
    th_ref_k = theta_wp + (theta_fc-theta_wp)*ETn;
    
    P_hor = P_k*ones(N,1);
    ETc_hor = ETc_k*ones(N,1);
    th_ref_hor = th_ref_k*ones(N,1);

    sol = solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg,...
                 'p',[xk;P_hor;ETc_hor;th_ref_hor]);
    
    w_opt = full(sol.x);
    Uopt = w_opt(N+2:end);
    uk = Uopt(1);

    xk = full(f(xk,uk,[P_k;ETc_k]));

    theta_hist(k)=xk;
    u_hist(k)=uk;
    theta_ref_hist(k)=th_ref_k;

    w0=w_opt;
end

%% -------------------------
% Gráficas
% --------------------------

time = 1:Tsim;
fs = 18;          % Tamaño de letra maestro
fs_tit = fs + 4;  
negro = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% --- Subplot 1: Humedad del suelo ---
subplot(3,1,1)
plot(t, theta_hist, 'b', 'LineWidth', 3); hold on;
plot(t, theta_ref_hist, 'r--', 'LineWidth', 2.5);

% Ajuste de etiquetas: 
% Punto Marchitez con 'LabelVerticalAlignment', 'bottom' para que el texto quede ABAJO de la línea
yline(theta_wp, 'k:', 'Punto Marchitez', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');

yline(theta_sat, 'k:', 'Saturación', 'FontSize', fs-2, 'FontWeight', 'bold', ...
    'LabelVerticalAlignment', 'top', 'LabelHorizontalAlignment', 'left');

% CLAVE: Bajamos el límite inferior a 0.05 para que el texto de abajo no choque con el marco
ylim([0.05, 0.50]); 

ylabel('\theta', 'FontSize', fs+4, 'Color', negro, 'FontWeight', 'bold');
title('Humedad del suelo', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

lgdd = legend('\theta','\theta_{ref}(t)', 'Location', 'northeast');
lgdd.FontSize = fs-2;
lgdd.TextColor = negro;
lgdd.Color = 'w';
lgdd.EdgeColor = negro;

% --- Subplot 2: Acción de control ---
subplot(3,1,2)
stairs(t, u_hist, 'k', 'LineWidth', 3); hold on;
yline(6, 'r:', 'Máx. Riego (6.0)', 'LineWidth', 2.5, 'FontSize', fs, 'FontWeight', 'bold', ...
    'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'bottom');

ylim([-0.5, 9]); 
ylabel('Riego (mm/día)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Acción de control MPC', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

% --- Subplot 3: Perturbaciones ---
subplot(3,1,3)
plot(t, ETc_hist, 'LineWidth', 3); hold on;
stairs(t, P_hist, 'LineWidth', 3);

ylim([-0.5, 11]); 
ylabel('mm/día', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
xlabel('Tiempo (días)', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
title('Perturbaciones climáticas', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
grid on;
set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, 'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

lgd = legend('ETc(t)', 'Precipitación', 'Location', 'northeast');
lgd.FontSize = fs-2;
lgd.TextColor = negro;
lgd.Color = 'w';
lgd.EdgeColor = negro;