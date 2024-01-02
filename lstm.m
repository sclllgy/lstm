%% LSTMによるLorenz方程式の時系列予測

close all;
clear;
clc;

%% 乱数発生器のシード固定

seed = 0;
rng(seed);

%% Lorenz方程式の時系列データの生成

T_train = 5000;  % 訓練データの長さ
T_test  = 1000;  % 検証データの長さ
transient = 0;  % トランジェントの長さ

x0 = [1.0, 1.0, 1.0];  % 初期値
dt = 0.02;  % 刻み幅
params = [10.0, 28.0, 8.0/3.0];

dynamics = Lorenz(x0, dt, params);
data = dynamics.Runge_Kutta(T_train + T_test);

%% 学習データの作成

U_train = data(1:T_train, :);
D_train = data(2:T_train+1, :);
U_test = data(1+T_train:T_train+T_test-1, :);
D_test = data(2+T_train:T_train+T_test, :);

%% データの標準化

mu = mean(U_train);
sigma = std(U_train);

U_train = (U_train - mu) ./ sigma;
D_train = (D_train - mu) ./ sigma;
U_test = (U_test - mu) ./ sigma;

%% LSTM Network の構築

Nu = 3;
Ny = 3;
Nh = 200;

layers = [sequenceInputLayer(Nu)
          lstmLayer(Nh)
          fullyConnectedLayer(Ny)
          regressionLayer];

%% 学習オプションの設定

opts = trainingOptions('adam', ...
                       'MaxEpochs', 200, ...
                       'GradientThreshold', 1, ...
                       'InitialLearnRate', 0.005, ...
                       'LearnRateSchedule', 'piecewise', ...
                       'LearnRateDropPeriod', 125, ...
                       'LearnRateDropFactor', 0.2, ...
                       'Verbose', 0, ...
                       'Plots', 'training-progress');

%% LSTM Network の学習

net = trainNetwork(U_train', D_train', layers, opts);

%% フリーラン予測

Y_pred = zeros(T_test-1, Ny);

% 過去データによる状態の更新
net = predictAndUpdateState(net, U_train');

% 1ステップ目の予測
[net, Y_pred(1,:)] = predictAndUpdateState(net, U_test(1,:)');

% 2ステップ目以降の予測
for i = 2:T_test-1
    [net, Y_pred(i,:)] = predictAndUpdateState(net, Y_pred(i-1,:)');
end

Y_pred = sigma .* Y_pred + mu;

%% グラフの描画

figure(1);
subplot(3,1,1); hold on;
plot(D_test(:,1), '-', LineWidth=2.0);
plot(Y_pred(:,1), '.-', LineWidth=2.0);
ylabel('$x(n)$', Interpreter='latex');
legend('Target','Predict', Interpreter='latex');
ax = gca;
ax.TickLabelInterpreter='latex';
set(ax, FontSize=16, YDir='reverse');
grid on;

subplot(3,1,2); hold on;
plot(D_test(:,2), '-', LineWidth=2.0);
plot(Y_pred(:,2), '.-', LineWidth=2.0);
xlabel('$n$', Interpreter='latex');
ylabel('$y(n)$', Interpreter='latex');
legend('Target','Predict', Interpreter='latex');
ax = gca;
ax.TickLabelInterpreter='latex';
set(ax, FontSize=16, YDir='reverse');
grid on;

subplot(3,1,3); hold on;
plot(D_test(:,3), '-', LineWidth=2.0);
plot(Y_pred(:,3), '.-', LineWidth=2.0);
xlabel('$n$', Interpreter='latex');
ylabel('$z(n)$', Interpreter='latex');
legend('Target','Predict', Interpreter='latex');
ax = gca;
ax.TickLabelInterpreter='latex';
set(ax, FontSize=16, YDir='reverse');
grid on;

%% RMSE（平均二乗誤差平方根）の算出

epsilon = 1.0;  % 許容誤差
for n = 1:T_test-1
    rmse_test = rmse(Y_pred(n,:), D_test(n,:));
    disp(['RMSE = ', num2str(rmse_test)]);
    if rmse_test > epsilon
        valid_time = n-1;
        disp(['valid time =', num2str(valid_time)]);
        break
    end
    if n==T_test-1
        valid_time = n;
        disp(['valid time =', num2str(valid_time)]);
    end
end

%% ネットワークの状態パラメータのリセット
net = resetState(net);

%% 予測

Y_pred = zeros(T_test-1, Ny);

% 過去データによる状態の更新
net = predictAndUpdateState(net, U_train');

% 2ステップ目以降の予測
for i = 1:T_test-1
    [net, Y_pred(i,:)] = predictAndUpdateState(net, U_test(i,:)');
end

Y_pred = sigma .* Y_pred + mu;

%% グラフの描画

figure(2);
subplot(3,1,1); hold on;
plot(D_test(:,1), '-', LineWidth=2.0);
plot(Y_pred(:,1), '.-', LineWidth=2.0);
ylabel('$x(n)$', Interpreter='latex');
legend('Target','Predict', Interpreter='latex');
ax = gca;
ax.TickLabelInterpreter='latex';
set(ax, FontSize=16, YDir='reverse');
grid on;

subplot(3,1,2); hold on;
plot(D_test(:,2), '-', LineWidth=2.0);
plot(Y_pred(:,2), '.-', LineWidth=2.0);
xlabel('$n$', Interpreter='latex');
ylabel('$y(n)$', Interpreter='latex');
legend('Target','Predict', Interpreter='latex');
ax = gca;
ax.TickLabelInterpreter='latex';
set(ax, FontSize=16, YDir='reverse');
grid on;

subplot(3,1,3); hold on;
plot(D_test(:,3), '-', LineWidth=2.0);
plot(Y_pred(:,3), '.-', LineWidth=2.0);
xlabel('$n$', Interpreter='latex');
ylabel('$z(n)$', Interpreter='latex');
legend('Target','Predict', Interpreter='latex');
ax = gca;
ax.TickLabelInterpreter='latex';
set(ax, FontSize=16, YDir='reverse');
grid on;

%% RMSE（平均二乗誤差平方根）の算出

epsilon = 1.0;  % 許容誤差
for n = 1:T_test-1
    rmse_test = rmse(Y_pred(n,:), D_test(n,:));
    disp(['RMSE = ', num2str(rmse_test)]);
    if rmse_test > epsilon
        valid_time = n-1;
        disp(['valid time =', num2str(valid_time)]);
        break
    end
    if n==T_test-1
        valid_time = n;
        disp(['valid time =', num2str(valid_time)]);
    end
end
