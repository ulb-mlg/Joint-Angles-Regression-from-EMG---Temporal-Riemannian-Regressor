# leap_smoother.py

import math, time
import numpy as np

# helpers
def wrap_pi(a):  # (-pi, pi]
    return (a + math.pi) % (2 * math.pi) - math.pi


class OneEuroFilter:
    
    # One-Euro filter homepage: https://gery.casiez.net/1euro/

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.t_prev = None
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, t, x):
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, 0.0
            return x
        dt = max(1e-4, t - self.t_prev)
        
        # low-pass derivative
        dx = (x - self.x_prev) / dt
        ad = self._alpha(self.d_cutoff, dt)
        dx_hat = ad * dx + (1 - ad) * self.dx_prev
        
        # adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat
    

class AngleUnwrapper:
    def __init__(self):
        self.last = None
        self.accum = 0.0

    def update(self, a):
        if self.last is None:
            self.last = a
            return a
        da = wrap_pi(a - self.last)
        self.accum += self.last + da - a
        self.last = a + self.accum
        return self.last

class SlewLimiter:
    def __init__(self, v_max=8.0, a_max=120.0):
        self.v_max = float(v_max)  # rad/s
        self.a_max = float(a_max)  # rad/s^2
        self.t = None
        self.x = None
        self.v = 0.0

    def update(self, t, x_target):
        if self.t is None:
            self.t, self.x, self.v = t, x_target, 0.0
            return x_target
        dt = max(1e-4, t - self.t)
        err = x_target - self.x
        v_des = max(-self.v_max, min(self.v_max, err / dt))
        dv = v_des - self.v
        a = max(-self.a_max, min(self.a_max, dv / dt))
        self.v += a * dt
        self.v = max(-self.v_max, min(self.v_max, self.v))
        self.x += self.v * dt
        self.t = t
        return self.x


class LeapSmoother:
    """
    Smoother to reduce jitters on the physical leap hand during operation 
    Call update(angles) each of the main running loop.

    Implemented smoothing:
      - angle unwrapping
      - 1€ filter (adaptive low pass)
      - velocity/acceleration limiting
      - tiny deadzone
      - optional: DIP<-PIP coupling (reduces fingertip jitter)
    
    To be implemented:
      - Interpolation smoothing
    """
    def __init__(self,
                 n=20,
                 min_cutoff=1.0, beta=0.35, d_cutoff=1.0,
                 v_max=8.0, a_max=120.0,
                 deadzone_deg=0.25,
                 max_speed_rad_per_s=15.0,
                 dip_from_pip=True, dip_ratio=0.66):
        self.n = n
        self.filters = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n)]
        self.unwrap = [AngleUnwrapper() for _ in range(n)]
        self.slew   = [SlewLimiter(v_max, a_max) for _ in range(n)]
        self.deadzone = math.radians(deadzone_deg)
        self.max_speed = float(max_speed_rad_per_s)
        self.last_out = [None]*n
        self.dip_from_pip = dip_from_pip
        self.dip_ratio = float(dip_ratio)

        # URDF indices for (PIP, DIP) pairs based on your mapping (no pinky on leap hand)
        self.pip_dip_pairs = [(2,3), (7,8), (12,13)] # , (17,18)]  

    def _apply_dip_pip_coupling(self, x):
        if not self.dip_from_pip: return x
        # DIP := r*PIP + (1-r)*DIP  (keeps a little independence)
        r = self.dip_ratio
        for pip, dip in self.pip_dip_pairs:
            x[dip] = r * x[pip] + (1.0 - r) * x[dip]
        return x

    def update(self, angles20, t=None):
        
        t = time.time() if t is None else float(t)
        x = list(angles20)

        # unwrap + spike clamp + 1€ filter
        for i in range(self.n):

            u = self.unwrap[i].update(x[i])

            # clamp extreme speed spikes before filtering
            prev = self.filters[i].x_prev if self.filters[i].x_prev is not None else u
            dt = (t - self.filters[i].t_prev) if self.filters[i].t_prev else 1/60.0
            if dt > 0:
                v = (u - prev)/dt
                if abs(v) > self.max_speed:
                    u = prev + self.max_speed * dt * (1 if v >= 0 else -1)
            xf = self.filters[i].update(t, u)
            
            # deadzone vs last output
            lo = self.last_out[i]
            if lo is not None and abs(xf - lo) < self.deadzone:
                xf = lo
            
            # slew-limit
            xs = self.slew[i].update(t, xf)
            x[i] = xs
            self.last_out[i] = xs
        
        # light biomechanical coupling between pip and dip
        x = self._apply_dip_pip_coupling(x)
        return np.array(x)
