import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# numerical integration fail
from func_timeout import func_timeout, FunctionTimedOut
import time


class HodgkinHuxley():
    """Hodgkin-Huxley model class"""

    def __init__(self):

        # membrane capacitance, in uF/cm^2
        self.C_m = 1.0
        self.init_parms()

    def init_parms(self):
        self.Em = -65.0
        self.gNa, self.gK, self.gL = 120.0, 36.0, 0.3
        # np.array for broadcasting in _V()
        self.ENa, self.EK, self.EL = self._V(np.array([-115.0, 12.0, -10.613]))

    def set_parms(self, max_conduct, rev_potential):
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

    def C_K(self, n):
        """
        potassim membrane current (in uA/cm^2)
        parameter: V, n
        """
        return self.gK * n**4

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
        return self.C_Na(m, h) * (V - self.ENa) \
            + self.C_K(n) * (V - self.EK) \
            + self.C_L() * (V - self.EL)

    def _I_inj(self, t, inputs):
        """
        input current
        """
        injected = 0
        for (t0, Vc) in inputs:
            injected += Vc*(t > t0) - Vc * (t > t0+1)

        return injected

    def _I_clamped(self, t, start, end, Vc):
        """
        input current
        """
        return Vc*(t > start) - Vc * (t > end) + self.Em

    @staticmethod
    def _equations_clamped(t, X, inputs, self):
        """
        integrate
        """
        m, h, n = X
        time.sleep(3)
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

    def simulation(self, sim_interval, clamped, inputs,
                   blocked_Na=False, blocked_K=False):
        """
        Simulate equation system via numeric integration. We use scipy's
        solve_ivp to solve an initial value problem for a system of ODEs.

        We use the RK23 (explicit Runge-Kutta method) method since it seams
        best suited for the Hodgkin-Huxley model in terms of speed and
        convergence.

        Parameters
        ----------
        sim_interval : tuple of int
            interval to compute the numerical integration
        clamped : bool
            clamped simulation if True, else injected current
        inputs : list of int (length factor of 3)
            list of [start, end, voltage] repeated varible times
        blocked_Na : bool (default: False)
            if True sodium channel is blocked
        blocked_K : bool (default: False)
            if True potassium channel is blocked

        Returns
        -------
        results : np.Array
            matrix of simulation results,
            dimension dependant on number of varibles to simulate
            (3 for clamped, 4 for injected) and number of time steps
            of the solver (see t_eval)
        """

        initial_values = self._get_initial_values()
        self.Em = initial_values[0]

        if clamped:
            # arguments passed to the respective equations function
            args_equations = [inputs, self]

            # respective equations function (clamped or injected)
            fct = self._equations_clamped

            # we don't need to simulate the voltage, since it is
            # controlled (clamped), only initial values for n, h, m
            initial_values = initial_values[-3:]
        else:
            args_equations = [inputs, blocked_Na, blocked_K, self]
            fct = self._equations_injected

        # arguments to be passed to solve_ivp as dictionary (kwargs)
        kwargs_ivp = {'method': 'RK23',
                      'vectorized': True,
                      't_eval': np.arange(sim_interval[0],
                                          sim_interval[1],
                                          0.2),
                      'args': args_equations
                      }

        # the ODE solver might run into an infinite loop or crashes,
        # therefore we set a timeout of 2 seconds
        try:
            result = func_timeout(2,
                                  solve_ivp,
                                  args=(fct,
                                        sim_interval,
                                        initial_values),
                                  kwargs=kwargs_ivp)
            return result

        except FunctionTimedOut:
            return 'Timeout: The ODE solver could not find a solution. ' \
                    'Please try a different parameter input.'

    def _get_initial_values(self):
        """
        Get initial values for solving the differential equations.
        Since the Hodgkin-Huxley model usually has a steady state,
        we can calculate the root of the equation system and use them
        as initial values.

        fsolve is a scipy function for finding the roots of a function.
        The initial values for the optimisation are simply the resting
        membrane potential and 0's for m, h, n.

        Note that we might run into numerical issues with exact initial
        values in the integration, so we round the results sufficiently.

        Returns
        -------
        results : np.Array (shape 4x1)
            optimisation results: initial values for diffeq
        """
        def _equations(X):
            V, m, h, n = X

            dVdt = -self._I_channel(V, m, h, n)
            dmdt = self._alpha_m(V)*(1.0-m) - self._beta_m(V)*m
            dhdt = self._alpha_h(V)*(1.0-h) - self._beta_h(V)*h
            dndt = self._alpha_n(V)*(1.0-n) - self._beta_n(V)*n

            return dVdt, dmdt, dhdt, dndt
        return np.round(fsolve(_equations, [self.Em, .0, .0, .0]), 2)
