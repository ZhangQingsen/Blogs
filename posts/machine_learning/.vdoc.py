# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: false
#| output: false

# Import necessary libraries
import qiskit
import numpy as np
from numpy import pi
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate, QFT, RYGate, MCXGate
from scipy.linalg import expm

# previous functions
def state_prep(mat_A, vec_b, n_c=1):
  N_b = vec_b.shape[0]
  n_b = np.round(np.log2(N_b))
  qr_b = QuantumRegister(n_b, name='b')  # b-register
  qr_c = QuantumRegister(n_c, name='c')  # c-register
  qr_a = QuantumRegister(1, name='a')    # a-register
  qc = QuantumCircuit(qr_c, qr_b, qr_a)  # complete circuit

  norm_b = vec_b / np.linalg.norm(vec_b)
  qc.initialize(norm_b, qr_b)
  return qc, qr_b, qr_c, qr_a

def phase_estimation(qc, qr_b, qr_c, mat_A, t):
    n_c = len(qr_c)
    
    # 1. Hadamard on c-register
    qc.h(qr_c)
    
    # 2. Apply controlled-U^(2^j) operations
    U = expm(1j * t * mat_A)  # 计算 e^(iAt)
    U_gate = UnitaryGate(U, label="e^{iAt}")  # 创建 U gate
  
    for j in range(n_c):
        power = 2**j  # 计算 U^(2^j)
        controlled_U_pow = U_gate.power(power).control(1)  # 控制操作
        qc.append(controlled_U_pow, [qr_c[j], *qr_b]) 
    
    # 3. Apply IQFT on c-register
    iqft = QFT(num_qubits=n_c, do_swaps=False).inverse()
    qc.append(iqft, qr_c)  # 作用于 c-register
    
    return qc

def controlled_rotation(qc, qr_c, qr_a, C, t, n_c):
    """
    实现受控旋转门，作用于 a-register，基于 c-register 的相位估计结果
    :param qc: 量子电路（需已经过相位估计到 |psi_4> 状态）
    :param qr_c: c-register（存储相位估计结果）
    :param qr_a: a-register（辅助比特）
    :param C: 缩放常数
    :param t: 时间参数（用于计算真实特征值）
    :param n_c: c-register 的量子比特数
    """
    # 遍历所有可能的相位估计结果 k（从1到2^n_c -1）
    for k in range(1, 2**n_c):
        # Step 1: 将二进制状态k转换为控制条件（设置X门）
        # 保存需要翻转的量子比特索引
        flip_qubits = []
        for i in range(n_c):
            if (k & (1 << i)) == 0:  # 如果第i位是0，需要翻转
                qc.x(qr_c[i])
                flip_qubits.append(i)
        
        # Step 2: 计算旋转角度 theta_j
        # 计算相位 phi（注意处理补码）
        phi = k / (2**n_c)
        if k > 2**(n_c - 1):
            phi -= 1.0  # 处理负相位（二进制补码）
        
        # 计算真实特征值 lambda_j = (2π / t) * phi
        lambda_j = (2 * np.pi / t) * phi
        
        # 计算旋转角度 theta_j = 2*arcsin(C / lambda_j)
        if abs(lambda_j) > 1e-8:  # 避免除以零
            theta_j = 2 * np.arcsin(C / lambda_j)
        else:
            theta_j = 0  # 若lambda_j太小，跳过旋转
        
        # Step 3: 应用多控制 RY 门到 a-register
        if abs(theta_j) > 1e-8:  # 仅当角度有效时应用
            ry_gate = RYGate(theta_j).control(num_ctrl_qubits=n_c, ctrl_state='1'*n_c)
            qc.append(ry_gate, qr_c[:] + [qr_a[0]])
        
        # Step 4: 恢复被翻转的量子比特
        for i in flip_qubits:
            qc.x(qr_c[i])

    return qc
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
N_b = 2 # number of variables (b.shape[0])
n_b = np.log2(N_b) # number of qubits in b-register
# n_b = 1 # N_b = 2^(n_b), n_b = np.log2(N_b)
n_c = 1 # number of qubits in c-register
N_c = 2 # N_c = 2^(n_c)
t = np.pi/8
C = 0.01
#
#
#
#
#
#
#
#
# prepare the previous steps
mat_A = np.array([[1, -1/3], [-1/3, 1]])
vec_b = np.array([0,1])
psi_1, qr_b, qr_c, qr_a = state_prep(mat_A, vec_b, n_c=1)
psi_4 = phase_estimation(psi_1, qr_b, qr_c, mat_A, t)
psi_6 = controlled_rotation(psi_4, qr_c, qr_a, C=C, t=t, n_c=n_c)

# psi_6.draw(output='mpl')
#
#
#
#
#
#
#
#
#
#
#
#
#
psi_7 = psi_6.copy()
psi_7.append(QFT(n_c, do_swaps=False), qr_c)

psi_7.draw(output='mpl')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
psi_8=psi_7.copy()
U_inv = expm(-1j * t * mat_A)
U_gate_inv = UnitaryGate(U_inv, label="e^{-iAt}")  # Inverse U gate


# Apply inverse controlled-U operations in reverse order
for j in reversed(range(n_c)):
    power = 2**j
    controlled_U_pow_inv = U_gate_inv.power(power).control(1)
    psi_8.append(controlled_U_pow_inv, [qr_c[j], *qr_b])

psi_8.draw(output='mpl')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
psi_9 = psi_8.copy()
psi_9.h(qr_c)
psi_9.draw(output='mpl')
# Do the measurement
c_b = ClassicalRegister(len(qr_b), name='cb')  # 存储 b-register 测量值
c_a = ClassicalRegister(1, name='ca')  # 存储 a-register 测量值
psi_9.add_register(c_b, c_a)  # 添加经典寄存器

psi_9.measure(qr_b, c_b)  # 测量 b-register 并存入 cb
psi_9.measure(qr_a, c_a)  # 测量 a-register 并存入 ca

psi_9.draw(output='mpl')

# 运行并查看结果
simulator = Aer.get_backend('statevector_simulator')
t_psi_9 = transpile(psi_9, simulator)
result = simulator.run(t_psi_9, shots=2 ** 21).result()
counts = result.get_counts(t_psi_9)
counts = result.get_counts(t_qc)
print(counts)

# 解析测量结果
# 假设 b-register 占据低位，a-register 占据最高位
measured_a1 = sum(counts[key] for key in counts if key[-1] == '1')
measured_a0 = sum(counts[key] for key in counts if key[-1] == '0')

# 计算归一化比值
if measured_a1 + measured_a0 > 0:
    print("P(a=1) =", measured_a1 / (measured_a1 + measured_a0))
    print("P(a=0) =", measured_a0 / (measured_a1 + measured_a0))

#
#
#
#
def phase_estimation(qc, qr_b, qr_c, mat_A, t):
    n_c = len(qr_c)
    
    # 1. Hadamard on c-register
    qc.h(qr_c)
    
    # 2. Apply controlled-U^(2^j) operations
    U = expm(1j * t * mat_A)  # 计算 e^(iAt)
    U_gate = UnitaryGate(U, label="e^{iAt}")  # 创建 U gate
  
    for j in range(n_c):
        power = 2**j  # 计算 U^(2^j)
        controlled_U_pow = U_gate.power(power).control(1)  # 控制操作
        qc.append(controlled_U_pow, [qr_c[j], *qr_b]) 
    
    # 3. Apply IQFT on c-register
    iqft = QFT(num_qubits=n_c, do_swaps=False).inverse()
    qc.append(iqft, qr_c)  # 作用于 c-register
    
    return qc

mat_A = np.array([[1, -1/3], [-1/3, 1]])
vec_b = np.array([0,1])
t = np.pi/8
psi_1, qr_b, qr_c, qr_a = state_prep(mat_A, vec_b, n_c=1)


psi_4 = phase_estimation(psi_1, qr_b, qr_c, mat_A, t)

psi_4.draw(output='mpl')
#
#
#
#
#
#
#
