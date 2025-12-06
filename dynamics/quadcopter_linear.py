import numpy as np
from scipy.linalg import expm

class QuadcopterLinearized:
    """
    State x = [x, y, z, φ, θ, ψ, vx, vy, vz, p, q, r]^T
    Input u = [T, τx, τy, τz]  (total thrust + roll/pitch/yaw torques)
    """

    def __init__(self, mass=0.027,         # kg  (Crazyflie approx)
                       Ixx=1.4e-5,         # inertia values
                       Iyy=1.4e-5,
                       Izz=2.17e-5,
                       g=9.81,
                       Ts=1/48):           # sample time for MPC (50 Hz)

        self.mass = mass
        self.Ixx  = Ixx
        self.Iyy  = Iyy
        self.Izz  = Izz
        self.g    = g
        self.Ts   = Ts

        # continuous-time matrices
        self.Ac = self._build_A()
        self.Bc = self._build_B()

        # discretive for MPC use
        self.A, self.B = self._discretize(self.Ac, self.Bc, Ts)

    def _build_A(self):
        """Returns continuous-time state matrix Ac (12x12)."""
        g = self.g
        A = np.zeros((12, 12))

        A[0,6] = 1;   A[1,7] = 1;   A[2,8] = 1
        A[3,9] = 1;   A[4,10] = 1;   A[5,11] = 1

        A[6,4] = g
        A[7,3] = -g

        return A

    def _build_B(self):
        """Returns continuous-time input matrix Bc (12x4)."""
        B = np.zeros((12,4))
        m, Ixx, Iyy, Izz = self.mass, self.Ixx, self.Iyy, self.Izz

        B[8,0]  = 1/m

        B[9,1]  = 1/Ixx
        B[10,2] = 1/Iyy
        B[11,3] = 1/Izz

        return B

    def _discretize(self, Ac, Bc, Ts):
        I = np.eye(Ac.shape[0])
        A_d = I + Ac * Ts
        B_d = Bc * Ts
        return A_d, B_d
