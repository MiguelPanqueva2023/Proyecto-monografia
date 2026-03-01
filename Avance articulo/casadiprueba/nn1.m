clc; clear; close all

%% ==========================
% DATASET
% ==========================
N = 100;

theta_k = 0.15 + 0.1*rand(N,1);
u_k     = 0.005*rand(N,1);
P_k     = 0.01*rand(N,1);
ET_k    = 0.002*rand(N,1);

theta_model = theta_k + u_k + P_k - ET_k;

theta_real = theta_model ...
           + 0.02*sin(5*theta_k) ...
           + 0.01*randn(N,1);

Y = (theta_real - theta_model)';   % residual (1 x N)
X = [theta_k u_k P_k ET_k]';        % inputs (4 x N)

%% ==========================
% NORMALIZACIÓN MANUAL
% ==========================
xmin = min(X,[],2);
xmax = max(X,[],2);
Xn = (X - xmin) ./ (xmax - xmin + eps);

ymin = min(Y);
ymax = max(Y);
Yn = (Y - ymin) / (ymax - ymin + eps);

%% ==========================
% ARQUITECTURA NN (3 CAPAS)
% 4 → 10 → 6 → 1
% ==========================
n_in  = 4;
n_h1  = 10;
n_h2  = 6;
n_out = 1;

rng(1)
W1 = 0.5*randn(n_h1,n_in);
b1 = zeros(n_h1,1);

W2 = 0.5*randn(n_h2,n_h1);
b2 = zeros(n_h2,1);

W3 = 0.5*randn(n_out,n_h2);
b3 = 0;

%% ==========================
% ENTRENAMIENTO (BACKPROP)
% ==========================
epochs = 2000;
lr = 0.01;

mse = zeros(epochs,1);

for e = 1:epochs

    % ----- FORWARD -----
    Z1 = W1*Xn + b1;
    A1 = tanh(Z1);

    Z2 = W2*A1 + b2;
    A2 = tanh(Z2);

    Yhat = W3*A2 + b3;    % salida lineal

    % ----- ERROR -----
    E = Yn - Yhat;
    mse(e) = mean(E.^2);

    % ----- BACKPROP -----
    dY = -2*E / N;

    dW3 = dY * A2';
    db3 = sum(dY,2);

    dA2 = W3' * dY;
    dZ2 = dA2 .* (1 - A2.^2);

    dW2 = dZ2 * A1';
    db2 = sum(dZ2,2);

    dA1 = W2' * dZ2;
    dZ1 = dA1 .* (1 - A1.^2);

    dW1 = dZ1 * Xn';
    db1 = sum(dZ1,2);

    % ----- UPDATE -----
    W3 = W3 - lr*dW3;
    b3 = b3 - lr*db3;

    W2 = W2 - lr*dW2;
    b2 = b2 - lr*db2;

    W1 = W1 - lr*dW1;
    b1 = b1 - lr*db1;
end

%% ==========================
% VALIDACIÓN
% ==========================
Yhat = W3*tanh(W2*tanh(W1*Xn + b1) + b2) + b3;
Y_pred = Yhat*(ymax - ymin) + ymin;

figure
plot(Y,'b','LineWidth',1.5); hold on
plot(Y_pred,'r--','LineWidth',1.5)
legend('Residual real','Residual NN')
xlabel('Muestra')
ylabel('Residuo')
grid on

%% ==========================
% USO DEL MODELO (PASO MPC)
% ==========================
theta_k = 0.2;
u_k = 0.003;
P_k = 0.001;
ET_k = 0.0005;

theta_model = theta_k + u_k + P_k - ET_k;

x = [theta_k; u_k; P_k; ET_k];
x_n = (x - xmin) ./ (xmax - xmin + eps);

a1 = tanh(W1*x_n + b1);
a2 = tanh(W2*a1 + b2);
r_n = W3*a2 + b3;

r = r_n*(ymax - ymin) + ymin;
theta_next = theta_model + r;

disp(['Theta modelo: ', num2str(theta_model)])
disp(['Theta corregido: ', num2str(theta_next)])