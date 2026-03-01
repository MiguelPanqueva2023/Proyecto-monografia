clc; clear; close all

%% ============================================================
% 1. Cargar dataset generado por el MPC
% ============================================================
load greybox_dataset.mat   % contiene Xdata (Nx4) y Ydata (Nx1)

X = Xdata';   % 4 x N
Y = Ydata';   % 1 x N
N = size(X,2);

%% ============================================================
% 2. Normalización
% ============================================================
xmin = min(X,[],2);
xmax = max(X,[],2);
Xn = (X - xmin) ./ (xmax - xmin + eps);

ymin = min(Y);
ymax = max(Y);
Yn = (Y - ymin) / (ymax - ymin + eps);

%% ============================================================
% 3. Arquitectura NN (4 → 10 → 6 → 1)
% ============================================================
n_in  = 4;
n_h1  = 10;
n_h2  = 6;
n_out = 1;

rng(1)
W1 = 0.5*randn(n_h1,n_in);   b1 = zeros(n_h1,1);
W2 = 0.5*randn(n_h2,n_h1);   b2 = zeros(n_h2,1);
W3 = 0.5*randn(n_out,n_h2);  b3 = 0;

%% ============================================================
% 4. Entrenamiento
% ============================================================
epochs = 3000;
lr = 0.01;
mse = zeros(epochs,1);

for e = 1:epochs

    % Forward
    Z1 = W1*Xn + b1;   A1 = tanh(Z1);
    Z2 = W2*A1 + b2;  A2 = tanh(Z2);
    Yhat = W3*A2 + b3;

    % Error
    E = Yn - Yhat;
    mse(e) = mean(E.^2);

    % Backprop
    dY = -2*E/N;

    dW3 = dY*A2';     db3 = sum(dY,2);
    dA2 = W3'*dY;     dZ2 = dA2.*(1 - A2.^2);
    dW2 = dZ2*A1';    db2 = sum(dZ2,2);
    dA1 = W2'*dZ2;    dZ1 = dA1.*(1 - A1.^2);
    dW1 = dZ1*Xn';    db1 = sum(dZ1,2);

    % Update
    W3 = W3 - lr*dW3;  b3 = b3 - lr*db3;
    W2 = W2 - lr*dW2;  b2 = b2 - lr*db2;
    W1 = W1 - lr*dW1;  b1 = b1 - lr*db1;
end

%% ============================================================
% 5. Validación
% ============================================================
Yhat = W3*tanh(W2*tanh(W1*Xn + b1) + b2) + b3;
Ypred = Yhat*(ymax - ymin) + ymin;

% --- Parámetros Visuales Identicos ---
fs = 18;          % Tamaño de letra maestro
fs_tit = fs + 4;  
negro = [0, 0, 0];

figure('Color','w', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% Gráfica con COLORES Y GROSORES del script MPC:
% Residuo real en AZUL ('b')
plot(Y, 'b', 'LineWidth', 3); hold on; 

% Predicción en ROJO PUNTEADO ('r--') como la referencia del MPC
plot(Ypred, 'r--', 'LineWidth', 2.5); 

% Títulos y Etiquetas con el mismo formato
title('Red neuronal residual', 'FontSize', fs_tit, 'Color', negro, 'FontWeight', 'bold');
xlabel('Muestra', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');
ylabel('Residuo', 'FontSize', fs, 'Color', negro, 'FontWeight', 'bold');

% Leyenda con el mismo estilo del MPC
lgd = legend('Residuo real', 'Red neuronal', 'Location', 'northeast');
lgd.FontSize = fs-2;
lgd.TextColor = negro;
lgd.Color = 'w';
lgd.EdgeColor = negro;

% Configuración de ejes (IDÉNTICA al set(gca...) del MPC)
grid on;
set(gca, 'Color', 'w', 'XColor', negro, 'YColor', negro, 'GridColor', negro, ...
    'GridAlpha', 0.2, 'FontSize', fs, 'LineWidth', 1.5);

% Ajuste de límites para limpieza visual
ylim([min(Y)-0.02, max(Y)+0.04]);

%% ============================================================
% 6. Guardar red entrenada
% ============================================================
save NN_residual.mat W1 b1 W2 b2 W3 b3 xmin xmax ymin ymax
