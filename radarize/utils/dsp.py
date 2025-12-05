#!/usr/bin/env python3

"""Helper functions for signal processing.
"""

import numpy as np
import cv2
from numba import njit, objmode

# flip_ods_phase: 是否翻转 ODS（接收端信号）的相位。
# flip_aop_phase: 是否翻转 AOP（发送端信号）的相位
def reshape_frame(frame, flip_ods_phase=False, flip_aop_phase=False):
    """Use this to reshape RadarFrameFull messages."""
    # 获取帧的 platform 信息，表示雷达所在的平台
    platform = frame.platform
    # 获取 ADC 输出格式 (adc_output_fmt)：获取 ADC 输出格式 (adc_output_fmt)：
    adc_output_fmt = frame.adc_output_fmt
    # frame.rx_phase_bias 是一个包含了接收端相位偏移的数组。
    # 它是一个复数的交替数组，frame.rx_phase_bias[0::2] 是实部，frame.rx_phase_bias[1::2] 是虚部。
    # 将这两个部分按位置配对，形成复数数组，作为 rx_phase_bias
    rx_phase_bias = np.array(
        [
            a + 1j * b
            for a, b in zip(frame.rx_phase_bias[0::2], frame.rx_phase_bias[1::2])
        ]
    )
    # 获取雷达的 chirp 数量
    n_chirps = int(frame.shape[0])
    # 接收/发射天线数
    rx = np.array([int(x) for x in frame.rx])
    n_rx = int(frame.shape[1])
    tx = np.array([int(x) for x in frame.tx])
    n_tx = int(sum(frame.tx))
    # 每个 chirp 中的样本数
    n_samples = int(frame.shape[2])

    # 
    return _reshape_frame(
        np.array(frame.data),
        platform,
        adc_output_fmt,
        rx_phase_bias,
        n_chirps,
        rx,
        n_rx,
        tx,
        n_tx,
        n_samples,
        flip_ods_phase=flip_ods_phase,
        flip_aop_phase=flip_aop_phase,
    )

# 实际进行数据立方体生成的内部函数
@njit(cache=True) # 装饰器加速
# Numba 提供的装饰器表示该函数会使用 JIT（即时编译）技术加速执行。
# cache=True 使得编译后的代码会被缓存，在下次调用时可以直接加载，避免重新编译，从而提高性能
def _reshape_frame(
    data,
    platform,
    adc_output_fmt,
    rx_phase_bias,
    n_chirps,
    rx,
    n_rx,
    tx,
    n_tx,
    n_samples,
    flip_ods_phase=False,
    flip_aop_phase=False,
):
    if adc_output_fmt > 0:
        # 原始数据 data 是交替存储的实部和虚部
        radar_cube = np.zeros(len(data) // 2, dtype=np.complex64)
        # radar_cube[0::2] 为每两个元素组合成一个复数（虚部乘以 1j）
        radar_cube[0::2] = 1j * data[0::4] + data[2::4]
        radar_cube[1::2] = 1j * data[1::4] + data[3::4]
        # 数据重塑成 (n_chirps, n_rx, n_samples)
        radar_cube = radar_cube.reshape((n_chirps, n_rx, n_samples))

        # Apply RX phase correction for each antenna.
        if "xWR68xx" in platform:
            if flip_ods_phase:  # Apply 180 deg phase change on RX2 and RX3
                # 对接收端的 RX2 和 RX3 天线进行 180 度相位翻转。
                # 遍历 rx 列表，对应天线开启时，进行相位翻转
                c = 0
                for i_rx, rx_on in enumerate(rx):
                    if rx_on:
                        if i_rx == 1 or i_rx == 2:
                            radar_cube[:, c, :] *= -1
                        c += 1
            elif flip_aop_phase:  # Apply 180 deg phase change on RX1 and RX3
                # 对接收端的 RX1 和 RX3 天线进行 180 度相位翻转
                c = 0
                for i_rx, rx_on in enumerate(rx):
                    if rx_on:
                        if i_rx == 0 or i_rx == 2:
                            radar_cube[:, c, :] *= -1
                        c += 1
        # 经过相位校正后，将数据重塑为 (n_chirps // n_tx, n_rx * n_tx, n_samples)，
        # 即根据发送天线数量将数据重新组织
        radar_cube = radar_cube.reshape((n_chirps // n_tx, n_rx * n_tx, n_samples))

        # Apply RX phase correction from calibration.
        # 对于每个有效的发送和接收天线，应用相位偏差 rx_phase_bias。
        # 通过 v_rx 来计算对应的接收天线的偏差，并将其应用到相应的雷达数据
        c = 0
        for i_tx, tx_on in enumerate(tx):
            if tx_on:
                for i_rx, rx_on in enumerate(rx):
                    if rx_on:
                        v_rx = i_tx * len(rx) + i_rx
                        # print(v_rx)
                        radar_cube[:, c, :] *= rx_phase_bias[v_rx]
                        c += 1

    else:
        # 直接将数据重塑为 (n_chirps // n_tx, n_rx * n_tx, n_samples)，并转换为复数类型
        radar_cube = data.reshape((n_chirps // n_tx, n_rx * n_tx, n_samples)).astype(
            np.complex64
        )
    # n_chirps // n_tx 表示每个发送天线的 chirp 数量
    # (n_rx * n_tx)：表示一个 chirp 对应的接收信号的总数，等于接收天线数和发送天线数的乘积
    # n_samples 每个 chirp 的采样数
    return radar_cube


def reshape_frame_tdm(frame, flip_ods_phase=False):
    """Use this to reshape RadarFrameFull messages."""

    platform = frame.platform
    adc_output_fmt = frame.adc_output_fmt
    rx_phase_bias = np.array(
        [
            a + 1j * b
            for a, b in zip(frame.rx_phase_bias[0::2], frame.rx_phase_bias[1::2])
        ]
    )

    n_chirps = int(frame.shape[0])
    rx = np.array([int(x) for x in frame.rx])
    n_rx = int(frame.shape[1])
    tx = np.array([int(x) for x in frame.tx])
    n_tx = int(sum(frame.tx))
    n_samples = int(frame.shape[2])

    return _reshape_frame_tdm(
        np.array(frame.data),
        platform,
        adc_output_fmt,
        rx_phase_bias,
        n_chirps,
        rx,
        n_rx,
        tx,
        n_tx,
        n_samples,
        flip_ods_phase=flip_ods_phase,
    )

# 将原始雷达数据的接收天线数据按照发射天线（n_tx）的数量重新组织，生成一个新的 radar_cube_tdm
# 每个发射天线的数据块与其对应的接收天线数据重新排列，从而实现时域复用
@njit(cache=True)
def _tdm(radar_cube, n_tx, n_rx):
    radar_cube_tdm = np.zeros(
        # 原本是n_chirps // n_tx, n_rx * n_tx, n_samples的[:, :8, :]
        # 现在第一维等于又乘回去
        (radar_cube.shape[0] * n_tx, radar_cube.shape[1], radar_cube.shape[2]),
        dtype=np.complex64,
    )

    for i in range(n_tx):
        # 遍历每个发射天线
        # 对 radar_cube_tdm 的第一维chirps进行间隔为 n_tx 的切片更新，即每 n_tx 行分配数据
        # 对 radar_cube_tdm 的第二维（天线）进行切片，选择对应的接收天线数据
        # 等式右边表示从原始 radar_cube 中提取对应发射天线（i）对应的接收天线数据
        radar_cube_tdm[i::n_tx, i * n_rx : (i + 1) * n_rx] = radar_cube[
            :, i * n_rx : (i + 1) * n_rx
        ]

    return radar_cube_tdm


@njit(cache=True)
def _reshape_frame_tdm(
    data,
    platform,
    adc_output_fmt,
    rx_phase_bias,
    n_chirps,
    rx,
    n_rx,
    tx,
    n_tx,
    n_samples,
    flip_ods_phase=False,
):

    radar_cube = _reshape_frame(
        data,
        platform,
        adc_output_fmt,
        rx_phase_bias,
        n_chirps,
        rx,
        n_rx,
        tx,
        n_tx,
        n_samples,
        flip_ods_phase,
    )

    radar_cube_tdm = _tdm(radar_cube, n_tx, n_rx)

    return radar_cube_tdm


@njit(cache=True)
def get_mean(x, axis=0):
    return np.sum(x, axis=axis) / x.shape[axis]


@njit(cache=True)
def cov_matrix(x):
    """Calculates the spatial covariance matrix (Rxx) for a given set of input data (x=inputData).
        Assumes rows denote Vrx axis.
    """

    _, num_adc_samples = x.shape
    x_T = x.T
    Rxx = x @ np.conjugate(x_T)
    Rxx = np.divide(Rxx, num_adc_samples)

    return Rxx


@njit(cache=True)
def gen_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    """Generate a steering vector for AOA estimation given the theta range, theta resolution, and number of antennas
        生成一个用于方向估计的引导向量
    """
    # 计算需要生成的引导向量的数量。
    # 如果 ang_est_range 为 90°，ang_est_resolution 为 1°，则总共需要 181 个方向来覆盖从 -90° 到 90° 的范围
    num_vec = (2 * ang_est_range + 1) / ang_est_resolution + 1
    num_vec = int(round(num_vec))
    steering_vectors = np.zeros((num_vec, num_ant), dtype="complex64")
    for kk in range(num_vec): # 遍历每个角度bin
        for jj in range(num_ant): # 遍历每个天线
            # 根据天线位置和估计的角度计算相位变化
            mag = (
                -1
                * np.pi
                * jj
                * np.sin((-ang_est_range - 1 + kk * ang_est_resolution) * np.pi / 180)
            )
            real = np.cos(mag) # 对应的实部是该相位变化的余弦值
            imag = np.sin(mag) # 对应的虚部是该相位变化的正弦值
            # 这两个值代表了该天线对该角度的贡献，通过 np.complex(real, imag) 将它们合成复数值并存储到 steering_vectors 中
            steering_vectors[kk, jj] = np.complex(real, imag)

    return (num_vec, steering_vectors)


@njit(cache=True)
def aoa_bartlett(steering_vec, sig_in):
    """
    Perform AOA estimation using Bartlett Beamforming on a given input signal (sig_in).
    """
    n_theta = steering_vec.shape[0]
    n_rx = sig_in.shape[1]
    n_range = sig_in.shape[2]
    y = np.zeros((sig_in.shape[0], n_theta, n_range), dtype="complex64")
    for i in range(sig_in.shape[0]):
        y[i] = np.conjugate(steering_vec) @ sig_in[i]
    return y


@njit(cache=True)
def aoa_capon(x, steering_vector):
    """
    Perform AOA estimation using Capon (MVDR) Beamforming on a rx by chirp slice
    基于 Capon (MVDR) Beamforming 算法的到达角度 (AOA) 估计实现
    MVDR（Minimum Variance Distortionless Response）是常用于方向估计的阵列处理方法，
    旨在从多个接收信号中估计信号源的方向，利用加权阵列数据以最小化信号的方差
    """

    # 计算输入信号 x 的协方差矩阵
    Rxx = cov_matrix(x)
    # 计算 Rxx 的逆矩阵，并将其转换为复数数据类型
    Rxx_inv = np.linalg.inv(Rxx).astype(np.complex64)
    # 将逆协方差矩阵 Rxx_inv 与引导向量的转置进行矩阵乘法，表示在指定角度的增益响应
    first = Rxx_inv @ steering_vector.T
    den = np.zeros(first.shape[1], dtype=np.complex64)
    # 获取引导向量的共轭复数
    steering_vector_conj = steering_vector.conj()
    first_T = first.T
    # 循环计算分母，每个元素是引导向量的共轭与 first_T 对应位置的元素的乘积累加结果
    for i in range(first_T.shape[0]):
        for j in range(first_T.shape[1]):
            den[i] += steering_vector_conj[i, j] * first_T[i, j]
    # 计算 den 的倒数，即 den 数组每个元素的倒数，这一步是为了在下一步中构建权重
    # 其同时也是copon功率谱
    den = np.reciprocal(den)
    # 通过矩阵乘法将 first 与 den 进行乘法，得到最终的权重 weights。
    # 这些权重是在 Capon Beamforming 算法下进行方向估计时需要用来加权不同接收信号的系数
    weights = first @ den

    return den, weights


@njit(cache=True)
def compute_range_azimuth(radar_cube, angle_res=1, angle_range=90, method="apes"):

    n_range_bins = radar_cube.shape[2] # 距离bin的数量
    n_rx = radar_cube.shape[1] # 接收天线数量
    n_chirps = radar_cube.shape[0] # # chirp 数量
    n_angle_bins = (angle_range * 2 + 1) // angle_res + 1 # 方位bin的数量

    range_cube = np.zeros_like(radar_cube)
    # 对 radar_cube 进行FFT，在这里，axis=2 表示对每个样本点进行 FFT（沿着每个 chirp 的采样维度）
    with objmode(range_cube="complex128[:,:,:]"):
        range_cube = np.fft.fft(radar_cube, axis=2)
    range_cube = np.transpose(range_cube, (2, 1, 0))
    range_cube = np.asarray(range_cube, dtype=np.complex64)
    # 创建一个与 range_cube 大小相同的空数组，用于存储每个范围的处理结果
    range_cube_ = np.zeros(
        (range_cube.shape[0], range_cube.shape[1], range_cube.shape[2]),
        dtype=np.complex64,
    )
    # 生成波束赋形矢量(引导向量)，用于后续到达角估计
    _, steering_vec = gen_steering_vec(angle_range, angle_res, n_rx)

    range_azimuth = np.zeros((n_range_bins, n_angle_bins), dtype=np.complex_)
    for r_idx in range(n_range_bins):
        # 对于每个距离bin，使用生成的波束赋形矢量对信号进行加权
        range_cube_[r_idx] = range_cube[r_idx]
        steering_vec_ = steering_vec
        # 方向估计只支持Capon算法
        if method == "capon":
            # 这里的第一个回参就是copon功率谱
            range_azimuth[r_idx, :], _ = aoa_capon(range_cube_[r_idx], steering_vec_)
        else:
            raise ValueError("Unknown method")
    # 幅值归一化
    range_azimuth = np.log(np.abs(range_azimuth))

    return range_azimuth

@njit(cache=True)
def compute_doppler_azimuth(
    radar_cube,
    angle_res=1,
    angle_range=90,
    range_initial_bin=0,
    range_subsampling_factor=2,
):

    n_chirps = radar_cube.shape[0]
    n_rx = radar_cube.shape[1]
    n_samples = radar_cube.shape[2]
    n_angle_bins = (angle_range * 2) // angle_res + 1

    # Subsample range bins.
    radar_cube_ = radar_cube[:, :, range_initial_bin::range_subsampling_factor]
    radar_cube_ -= get_mean(radar_cube_, axis=0)

    # Doppler processing.
    doppler_cube = np.zeros_like(radar_cube_)
    with objmode(doppler_cube="complex128[:,:,:]"):
        doppler_cube = np.fft.fft(radar_cube_, axis=0)
        doppler_cube = np.fft.fftshift(doppler_cube, axes=0)
    doppler_cube = np.asarray(doppler_cube, dtype=np.complex64)

    # Azimuth processing.
    _, steering_vec = gen_steering_vec(angle_range, angle_res, n_rx)

    doppler_azimuth_cube = aoa_bartlett(steering_vec, doppler_cube)
    # doppler_azimuth_cube = doppler_azimuth_cube[:,:,::5]
    doppler_azimuth_cube -= np.expand_dims(
        get_mean(doppler_azimuth_cube, axis=2), axis=2
    )

    doppler_azimuth = np.log(get_mean(np.abs(doppler_azimuth_cube) ** 2, axis=2))

    return doppler_azimuth


def normalize(data, min_val=None, max_val=None):
    """
    Normalize floats to [0.0, 1.0].
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    img = (((data - min_val) / (max_val - min_val)).clip(0.0, 1.0)).astype(data.dtype)
    return img

def preprocess_1d_radar_1843(
    radar_cube,
    angle_res=1,
    angle_range=90,
    range_subsampling_factor=2,
    min_val=10.0,
    max_val=None,
    resize_shape=(48, 48),
):
    """
    Turn radar cube into 1d doppler-azimuth heatmap.
    """

    heatmap = compute_doppler_azimuth(
        radar_cube,
        angle_res,
        angle_range,
        range_subsampling_factor=range_subsampling_factor,
    )

    heatmap = normalize(heatmap, min_val=min_val, max_val=max_val)

    heatmap = cv2.resize(heatmap, resize_shape, interpolation=cv2.INTER_AREA)

    return heatmap

