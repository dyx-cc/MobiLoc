%%% 对幅度和相位进行去噪和校准，合并为CSI

function csi_data = CombineCSI(amplitudes, phases)
    N = 6;
    freq = 60.48e9;
    c = physconst('LightSpeed');
    lambda = c/freq;
    d = lambda*0.58;

    load('antennas_mikrotik.mat');
    load('oscillator.mat');

    [num_samples, num_subcarriers] = size(amplitudes);
    pre_channel = zeros(6, 6, num_samples);

    % 只使用前30个天线返回的值
    % for jj=1:(num_subcarriers-2)
    % 使用32根天线返回的值
    for jj=1:num_subcarriers
        p = phases(:, jj);
        a = amplitudes(:, jj);
        p = p*2*pi/1024;
        p = exp(1i*p);

        converging_limit = 50;
        converging_retries = 0;
        converged = 0;
        % 若由于种子随机，初始时非收敛：
        while converged == 0
            try
                [p_1, phase_offset_0, converged] = Sanitize(p);
            catch
                disp(['Converging error on CSI'])
            end
            if converging_retries == converging_limit
                break
            end
            converging_retries = converging_retries+1;
        end

        if converging_retries == converging_limit
            disp(['Converging threshold reached, ignoring CSI'])
            continue
        end

        % 去除oscilator影响
        p = p/exp(1i*antenna_oscilator_phases(antenna_positions == jj));
        %a = amplitudes(:, jj).* p; % 不确定

        [row, col] = find(antenna_positions == jj);
        pre_channel(row, col, :) = a.*p(1:num_samples, :);
    end
    csi_data = reshape(pre_channel, [N*N, num_samples])';