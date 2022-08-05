import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from func_timeout import func_timeout, FunctionTimedOut


class HodgkinHuxley():
    """
    Hodgkin-Huxley model class

    Holds the model parameter and functions to calculate (simulation via
    numerical integration) the model with given inputs.
    """

    def __init__(self):
        """
        Initilize the Hodgkin-Huxley model class with the initial parameters.
        """
        # lipid bilayer capacitance is assumed to be at 1
        self.c_m = 1.0
        self.init_parms()

    def init_parms(self):
        """
        Initilize parameters according to Hodgkin and Huxley's original paper.
        """
        # resting membrane potential
        self.e_m = -65.0

        # conductance
        self.g_na, self.g_k, self.g_leak = 120.0, 36.0, 0.3

        # reversal potentials
        self.e_na, self.e_k, self.e_leak = 50.0, -77.0, -54.387

    def set_parms(self, max_conduct, rev_potential):
        """
        Helper function for setting the parameters.
        """
        self.g_na, self.g_k, self.g_leak = max_conduct
        self.e_na, self.e_k, self.e_leak = rev_potential

    def reset_parms(self):
        """
        Helper function to make name more expressive.
        """
        self.init_parms()

    def _alpha_n(self, V):
        """
        Calculate alpha_n function for gating kinetics, helper function
        of membrane voltage.

        Parameters
        ----------
        V : float
            membrane voltage

        Returns
        -------
        result : float
            value
        """
        return .01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def _beta_n(self, V):
        """
        Calculate beta_n function for gating kinetics, helper function
        of membrane voltage.

        Parameters
        ----------
        V : float
            membrane voltage

        Returns
        -------
        result : float
            value
        """
        return .125 * np.exp(-(V + 65.0) / 80.0)

    def _alpha_h(self, V):
        """
        Calculate alpha_h function for gating kinetics, helper function
        of membrane voltage.

        Parameters
        ----------
        V : float
            membrane voltage

        Returns
        -------
        result : float
            value
        """
        return .07 * np.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V):
        """
        Calculate beta_h function for gating kinetics, helper function
        of membrane voltage.

        Parameters
        ----------
        V : float
            membrane voltage

        Returns
        -------
        result : float
            value
        """
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def _alpha_m(self, V):
        """
        Calculate alpha_m function for gating kinetics, helper function
        of membrane voltage.

        Parameters
        ----------
        V : float
            membrane voltage

        Returns
        -------
        result : float
            value
        """
        return .1 * (V + 40.0)/(1.0 - np.exp(-(V + 40.0) / 10.0))

    def _beta_m(self, V):
        """
        Calculate beta_m function for gating kinetics, helper function
        of membrane voltage.

        Parameters
        ----------
        V : float
            membrane voltage

        Returns
        -------
        result : float
            value
        """
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def cond_na(self, m, h):
        """
        Calculate sodium membrane conductance (in uA/cm^2)

        Parameters
        ----------
        m : float
            probability for sodium channel subunit activation
        h : float
            probability for sodium channel subunit inactivation

        Returns
        -------
        result : float
            sodium conductance
        """
        return self.g_na * np.power(m, 3) * h

    def cond_k(self, n):
        """
        Calculate potassim membrane conductance (in uA/cm^2)

        Parameters
        ----------
        n : float
            probability for potassium channel subunit activation

        Returns
        -------
        result : float
            potassium conductance
        """
        return self.g_k * np.power(n, 4)

    #  Leak
    def cond_leak(self):
        """
        leak membrane conductance (in uA/cm^2)

        Returns
        -------
        result : float
            leak conductance
        """
        return self.g_leak

    def _I_channel(self, V, m, h, n):
        """
        Calculate the membrane current due to ion channels (Na+, K+ and Leak)
        (in uA/cm^2)

        Parameters
        ----------
        V : float
            membrane voltage
        m : float
            probability for sodium channel subunit activation
        h : float
            probability for sodium channel subunit inactivation
        n : float
            probability for potassium channel subunit activation

        Returns
        -------
        result : float
            membrane current due to ion channels
        """
        return self.cond_na(m, h) * (V - self.e_na) \
            + self.cond_k(n) * (V - self.e_k) \
            + self.cond_leak() * (V - self.e_leak)

    def _I_inj(self, t, inputs):
        """
        Calculate the injected voltage at time t.

        Parameters
        ----------
        t : float
            time varible
        inputs : list of int (3*n,)
            list of [start, end, voltage] repeated varible times

        Returns
        -------
        results : float
            clamped current at t
        """
        injected = 0
        for (t0, t1, vol) in inputs:
            injected += vol * (t > t0) - vol * (t > t1)

        return injected

    def _I_clamped(self, t, inputs):
        """
        Calculate the clamped voltage at time t.

        Parameters
        ----------
        t : float
            time varible
        inputs : list of int (3,)
            list of [start, end, voltage] repeated varible times

        Returns
        -------
        results : float
            clamped current at t
        """
        t0, t1, vol = inputs
        return vol * (t > t0) - vol * (t > t1) + self.e_m

    @staticmethod
    def _equations_clamped(t, vars, inputs, self):
        """
        Set of different equations to be integrated for simulation of
        clamped voltage. Note that the clamped and injected equations
        are in seperate functions for better readability.

        Parameters
        ----------
        t : float
            time varible
        vars : array-like, only called as ndarray of floats (3,)
            holds varibles for m, h and n
        inputs : list of int (3*n,)
            list of [start, end, voltage] repeated varible times

        Returns
        -------
        results : ndarray, shape (3,)
            results for the differentials at t
        """
        m, h, n = vars

        V = self._I_clamped(t, inputs)
        dmdt = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m
        dhdt = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h
        dndt = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n
        return np.array([dmdt, dhdt, dndt], dtype=object)

    @staticmethod
    def _equations_injected(t, vars, inputs, blocked_Na, blocked_K, self):
        """
        Set of different equations to be integrated for simulation of
        injected current.

        Parameters
        ----------
        t : float
            time varible
        vars : array-like, only called as ndarray of floats (4,)
            holds varibles for voltage, m, h and n
        inputs : list of int (3*n,)
            list of [start, end, voltage] repeated varible times
        blocked_Na : bool (default: False)
            if True sodium channel is blocked
        blocked_K : bool (default: False)
            if True potassium channel is blocked

        Returns
        -------
        results : ndarray, shape (4,)
            results for the differentials at t
        """

        V, m, h, n = vars

        dVdt = (self._I_inj(t, inputs) - self._I_channel(V, m, h, n)) / self.c_m

        if blocked_Na:
            dmdt = 0
        else:
            dmdt = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m

        dhdt = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h

        if blocked_K:
            dndt = 0
        else:
            dndt = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n

        return np.array([dVdt, dmdt, dhdt, dndt], dtype=object)

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
        results : Bunch object (fields 't' and 'y' of type ndarray)
            object of which we later use the fields 't' and 'y',
            't' time steps, 'y' matrix of simulation results:
            dimension dependant on number of varibles to simulate
            (3 for clamped, 4 for injected) and number of time steps
            of the solver (see t_eval)
        """

        initial_values = self._get_initial_values()
        self.e_m = initial_values[0]

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

            # here we need to tweak the model because with one of the ion
            # channels, the numerical intergration is unstable:
            # so we set different initial values
            if blocked_Na or blocked_K:
                initial_values = [-65.0, 0.05, 0.6, 0.32]

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
        results : ndarray, shape (4,)
            optimisation results: initial values for diffeq
        """
        def _equations(X):
            V, m, h, n = X

            dVdt = -self._I_channel(V, m, h, n)
            dmdt = self._alpha_m(V)*(1.0 - m) - self._beta_m(V) * m
            dhdt = self._alpha_h(V)*(1.0 - h) - self._beta_h(V) * h
            dndt = self._alpha_n(V)*(1.0 - n) - self._beta_n(V) * n

            return dVdt, dmdt, dhdt, dndt
        return np.round(fsolve(_equations, [self.e_m, .0, .0, .0]), 2)
