classdef squeezeLayer < nnet.layer.Layer % & nnet.layer.Formattable (Optional)
%% 重命名为transposeLayer
    properties
        % (Optional) Layer properties.

        % Declare layer properties here.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
    end

    properties (State)
        % (Optional) Layer state parameters.

        % Declare state parameters here.
    end

    properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state.

        % Declare nested networks with learnable and state parameters here.
    end

    methods
        function layer = squeezeLayer(name)
        %%重命名层名
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.Name = name;
            %%层说明，会在分析的时候显示
            layer.Description = "squeeze";

            % Define layer constructor function here.
        end

        function [Z] = predict(layer,X)
            Z = permute(X,[1,4,3,2]);%%转置就可以了
        end

       
    end
end
