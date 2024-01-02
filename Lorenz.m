classdef Lorenz < handle
    properties
        x0     % Initial State: [1 x 3]
        x      % Current State: [1 x 3]
        dt
        param  % [sigma, rho, beta]: [1 x 3]
    end

    methods
        % Constructor
        function obj = Lorenz(x0, dt, param)
            obj.x0 = x0;
            obj.x = x0;
            obj.dt = dt;
            obj.param = param;
        end
        % Runge-Kutta (RK4) method
        function X = Runge_Kutta(obj, n)
            X = zeros(n,3);
            X(1,:) = obj.x;
            dxdt = @(x) [obj.param(1)*(x(2)-x(1)), ...
                         x(1)*(obj.param(2)-x(3)) - x(2), ...
                         x(1)*x(2) - obj.param(3)*x(3)];
            for i = 2:n
                k1 = obj.dt * dxdt(obj.x);
                k2 = obj.dt * dxdt(obj.x + k1/2);
                k3 = obj.dt * dxdt(obj.x + k2/2);
                k4 = obj.dt * dxdt(obj.x + k3);
                X(i,:) = obj.x + (k1 + 2*k2 + 2*k3 + k4)/6;
                obj.x = X(i,:);
            end
        end
    end
end