%%% 根据CSI和FTM估计AoA和distance

% csi_data: CSI数据
% ftm_times: FTM时间戳
% ftm_order: FTM时间戳与CSI对应
% N: 天线阵列每边天线的个数
% d: 天线与相邻天线的间距
% freq: 设备采样频率

function [az, el, spectrum] = EstimateAoA(csi_data, L, W, d, freq, mode)
    
    % 2D-MUSIC算法
    az = zeros(0,1);
    el = zeros(0,1);

    if mode
        spectrum = zeros(31,181,0);
    end

    num_samples = length(csi_data);
    array = phased.URA('Size', [L W], 'ElementSpacing', [d d]);
    estimator = phased.MUSICEstimator2D('SensorArray', array,...
        'OperatingFrequency', freq,...
        'NumSignalsSource','Property',...
        'DOAOutputPort', true,...
        'AzimuthScanAngles', -91:1:91,...
        'ElevationScanAngles', -16:1:16);

    if mode == 'all'
        % 根据全部csi计算唯一角度
        [sp, doa] = estimator(csi_data(:,:));
        az = doa(1,:);
        el = doa(2,:);
        spectrum = sp(2:32,2:182);
    else
        mode = mode - 1;
        for i=1:num_samples-mode
            [sp, doa] = estimator(csi_data(i:i+mode,:));
            az(i,1) = doa(1);
            el(i,1) = doa(2);
            spectrum(:,:,i) = sp(2:32,2:182);
        end
    end