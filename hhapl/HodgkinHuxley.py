import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Tuple, List

# numerical integration fail
import signal


class HodgkinHuxley():
    """Hodgkin-Huxley model class"""

    def __init__(self):
        
        # membrane capacitance, in uF/cm^2
        self.C_m = 1.0
        self.init_parms()
        
    def init_parms(self):
        self.Em = -65.0
        self.gNa, self.gK, self.gL = 120.0, 36.0, 0.3
        self.ENa, self.EK, self.EL = self._V(np.array([-115.0, 12.0, -10.613]))
        
    def set_parms(self, max_conduct: List[float], rev_potential: List[float]):
        self.gNa, self.gK, self.gL = max_conduct
        self.ENa, self.EK, self.EL = self._V(rev_potential)
        
    def reset_parms(self):
        self.init_parms()
        
    def _V(self, Vm):
        return self.Em - Vm

    def _alpha_n(self, V):
        """
        gating kinetics,
        parameter: of membrane voltage
        """

        return .01 * ((self._V(V)+10) / (np.exp((self._V(V)+10)/10)-1))

    def _beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        """
        gating kinetics,
        parameter: of membrane voltage
        """

        return .125*np.exp(self._V(V)/80)

    def _alpha_h(self, V):
        """
        gating kinetics,
        parameter: of membrane voltage
        """

        return .07*np.exp(self._V(V)/20)

    def _beta_h(self, V):
        """
        gating kinetics,
        parameter: of membrane voltage
        """

        return 1 / (np.exp((self._V(V)+30)/20) + 1)

    def _alpha_m(self, V):
        """
        gating kinetics,
        parameter: of membrane voltage
        """
        return .1*((self._V(V)+25) / (np.exp((self._V(V)+25)/10)-1))

    def _beta_m(self, V):
        """
        gating kinetics,
        parameter: of membrane voltage
        """

        return 4 * np.exp(self._V(V)/18)

    def C_Na(self, m, h):
        """
        sodium membrane current (in uA/cm^2)
        parameter: V, m, h
        """
        return self.gNa * m**3 * h
        #return 0

    def C_K(self, n):
        """
        potassim membrane current (in uA/cm^2)
        parameter: V, n
        """
        return self.gK * n**4
        #return 0

    #  Leak
    def C_L(self):
        """
        leak membrane current (in uA/cm^2)
        parameter: V
        """
        return self.gL
    
    def _I_channel(self, V, m, h, n):
        """
        membrane current due to ion channels (Na+, K+ and Leak) (in uA/cm^2)
        parameter: V
        """
        return self.C_Na(m, h) * (V - self.ENa) + self.C_K(n) * (V - self.EK) + self.C_L() * (V - self.EL)

    def _I_inj(self, t, inputs):
        """
        input current
        """
        injected = 0
        for (t0, Vc) in inputs:
            injected += Vc*(t>t0) - Vc*(t>t0+1)
            
        return injected
    
    def _I_clamped(self, t, start, end, Vc):
        """
        input current
        """
        return Vc*(t>start) - Vc*(t>end) + self.Em

    @staticmethod
    def _equations_clamped(t, X, inputs, self):
        """
        integrate
        """
        m, h, n = X
        start, end, Vc = inputs

        V = self._I_clamped(t, start, end, Vc)
        dmdt = self._alpha_m(V)*(1.0-m) - self._beta_m(V)*m
        dhdt = self._alpha_h(V)*(1.0-h) - self._beta_h(V)*h
        dndt = self._alpha_n(V)*(1.0-n) - self._beta_n(V)*n
        return dmdt, dhdt, dndt
    
    @staticmethod
    def _equations_injected(t, X, inputs, blocked_Na, blocked_K, self):
        """
        integrate
        """
        
        V, m, h, n = X
        
        dVdt = (self._I_inj(t, inputs) - self._I_channel(V, m, h, n)) / self.C_m
        
        if blocked_Na:
            dmdt = 0
        else:
            dmdt = self._alpha_m(V)*(1.0-m) - self._beta_m(V)*m
            
        dhdt = self._alpha_h(V)*(1.0-h) - self._beta_h(V)*h
        
        if blocked_K:
            dndt = 0
        else:
            dndt = self._alpha_n(V)*(1.0-n) - self._beta_n(V)*n
        
        return dVdt, dmdt, dhdt, dndt

    def simulation(self, sim_intervall: Tuple[int, int], clamped: bool, inputs: List[int], blocked_Na = False, blocked_K = False):
        """
        simulate equation system via numeric integration
        Solve an initial value problem for a system of ODEs.
        """
        
        initial_values = self._get_initial_values()
        self.Em = initial_values[0]
        
        def handle_timeout(signum, frame):
            raise TimeoutError

        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(5)  # 5 seconds

        try:
            if clamped:
                X = solve_ivp(self._equations_clamped, sim_intervall, initial_values[-3:], args=(inputs, self), 
                              method='RK23', vectorized=True, t_eval=np.arange(sim_intervall[0], sim_intervall[1], 0.2))
            else:
                X = solve_ivp(self._equations_injected, (0, 50), initial_values, args=(inputs, blocked_Na, blocked_K, self),
                              method='RK23', vectorized=True, t_eval=np.arange(sim_intervall[0], sim_intervall[1], 0.2))
        except TimeoutError:
            # TODO msg into notebook
            print("The ODE solver could not find a solution. Please try a different parameter input.")
            return
        finally:
            signal.alarm(0)
            
        return X
    
    def _get_initial_values(self):
        """
        exact initial values break the numeric integration
        """
        def _equations(X):
            V, m, h, n = X

            dVdt = -self._I_channel(V, m, h, n)
            dmdt = self._alpha_m(V)*(1.0-m) - self._beta_m(V)*m
            dhdt = self._alpha_h(V)*(1.0-h) - self._beta_h(V)*h
            dndt = self._alpha_n(V)*(1.0-n) - self._beta_n(V)*n

            return dVdt, dmdt, dhdt, dndt
        return np.round(fsolve(_equations, [self.Em, .0, .0, .0]), 2)
    
        
