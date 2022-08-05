import numpy as np
import time

# ipywidgets as interactive Widgets for the Jupyter notebooks (and Voila)
from ipywidgets import Layout, VBox, Button, Dropdown, BoundedIntText

# bqplot as plotting library for Jupyter notebooks
import bqplot.pyplot as plt
from bqplot.interacts import BrushIntervalSelector
from bqplot import Label, LinearScale, Axis, Lines, Figure

# project classes
from hhapl import HodgkinHuxley

# Tweak for working with ipywidgets
# https://github.com/jupyter-widgets/ipywidgets/issues/2103
import functools

class Plots(HodgkinHuxley.HodgkinHuxley):
    """
    Plot class for the Hodgkin-Huxley action potential lab
    
    - generates all plots displayed in the application
    - initilizes bqplot figures and interactive Widgets
    - interacts with user input and manages the animations
    """

    def __init__(self):
        """
        Initilizing the plot class and inheriting Hodgkin-Huxley model class
        """
        super().__init__()
        
        #  standard parameters
        self.interval = (0, 20)
        self.clamped = True
        self.start = 5
        self.end = 15
        
    def calc(self, Vc):
        # TODO redundant function
        X = self.simulation(self.interval, self.clamped, [self.start, self.end, Vc], '')
        if isinstance(X, str):
            print(X)
        V = Vc * np.logical_and(X.t > self.start, X.t < self.end)
        
        return V, X

        
    def init_figure(self, width, height, label, xlim, ylim, num_lines, ap_labels = False):
        """
        Initilize figure
        
        bqplot allows only to change (adding elements such as marks) the current figure,
        but we are mostly dealing with multiple-figure-plots and since subplots are not
        available in bqplot, we pre-register or initilize all elements (marks). We do
        this to have a consistent order to later animate or manipulate specific marks.
        
        Parameters
        ----------
        width : int
            figure width
        height : int
            figure height
        width : int
            figure width
        label : string
            label corresponding to marks (vline, hline, e.g. 'Threshold')
        xlim : tuple of ints
            x limits of the current axes
        ylim : tuple of ints
            y limits of the current axes
        num_lines : int
            number of lines added to the figure
        ap_labels : bool
            if True, add predefined marks (vline, hline, label) to the figure
            in order to highlight the stages of an action potential (ap)           
            
        Returns
        -------
        fig : bqplot.figure.Figure object
            figure object
        """
        fig = plt.figure()
        
        axes_options = {
            "color": {"visible": False},
        }
        
        # TODO better colors
        colors = [['blue'],['green'],['red'],['orange']]
        
        # set the number of lines, values to be set on runtime (animation)
        # [0] to fixate certain color to certain line ([] would not work)
        for i in range(num_lines):
            plt.plot([0], [0], colors=colors[i], axes_options=axes_options)
            # TODO legend for conductance
            
        # TODO for
        if ap_labels:
            plt.hline(level=-55, line_style='dashed', stroke_width=1, label='Threshold')
            plt.label(text=['Threshold'], x=[0], y=[-50], colors=['black'], align='start', default_size=10, rotate_angle = 0)
            
            plt.vline(level=-50, line_style='dashed', stroke_width=1, label='Depolarisation')
            plt.label(text=['Depolarisation'], x=[-50], y=[0], colors=['black'], align='start', default_size=12, rotate_angle = -90)
            
            plt.vline(level=-50, line_style='dashed', stroke_width=1, label='Repolarisation')
            plt.label(text=['Repolarisation'], x=[-50], y=[0], colors=['black'], align='middle', default_size=12, rotate_angle = 0)
            
            plt.vline(level=-50, line_style='dashed', stroke_width=1, label='Hyperpolarisation')
            plt.label(text=['Hyperpolarisation'], x=[-50], y=[0], colors=['black'], align='start', default_size=12, rotate_angle = 0)
        
        # TODO compact
        # TODO ylabel only on the last plot
        # set plot properties
        plt.xlabel('Time (ms)')
        plt.ylabel(label)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        fig.layout.width = width
        fig.layout.height = height
        fig.layout.padding = '0px'
        fig.fig_margin={'top':10, 'bottom':35, 'left':50, 'right':50}
        
        fig.legend=False
        
        return fig
    
    def plot_Na(self):
        """
        calculate and plot
        """
        
        fig_voltage = self.init_figure('400px', '200px', 'Clamped voltage', self.interval, (-1, 100), 4)
        fig_conductance = self.init_figure('400px', '200px', 'Conductance', self.interval, (-1, 20), 4)
        
        # TODO for
        b20 = Button(description='20 mV')
        b40 = Button(description='40 mV')
        b60 = Button(description='60 mV')
                
        def play(b, Vc):
            self.reset_parms()
            V, X = self.calc(Vc)
            m, h, _ = X.y
            na_conductance = self.C_Na(m, h)
            
            # TODO
            match Vc:
                case 20:
                    l_V = fig_voltage.marks[0]
                    l_C = fig_conductance.marks[0]
                case 40:
                    l_V = fig_voltage.marks[1]
                    l_C = fig_conductance.marks[1]
                case 60:
                    l_V = fig_voltage.marks[2]
                    l_C = fig_conductance.marks[2]
                case _:
                    l_V = fig_voltage.marks[3]
                    l_C = fig_conductance.marks[3]
                        
            for i in range(len(V)):
                time.sleep(0.025)
                
                with l_V.hold_sync(), l_C.hold_sync():
                    l_V.x = X.t
                    l_V.y = V[:i]
                    l_C.x = X.t
                    l_C.y = na_conductance[:i]

        b20.on_click(functools.partial(play, Vc=20))
        b40.on_click(functools.partial(play, Vc=40))
        b60.on_click(functools.partial(play, Vc=60))
                    
        display(VBox([b20, b40, b60, fig_conductance, fig_voltage], layout=Layout(align_items='center', width='500px')))

    def plot_K(self):
        """
        calculate and plot
        """
        
        # labels: MathJax not supported with bqplot
        fig_voltage = self.init_figure('400px', '200px', 'Clamped voltage', self.interval, (-1, 100), 4)
        fig_C_Na = self.init_figure('400px', '200px', 'Conductance (Na)', self.interval, (-1, 20), 4) 
        fig_C_K = self.init_figure('400px', '200px', 'Conductance (K)', self.interval, (-1, 50), 4)
        
        b20 = Button(description='20 mV')
        b40 = Button(description='40 mV')
        b60 = Button(description='60 mV')
                
        def play(b, Vc):
            self.reset_parms()
            V, X = self.calc(Vc)
            m, h, n = X.y
            na_conductance = self.C_Na(m, h)
            k_conductance = self.C_K(n)
            
            # TODO
            match Vc:
                case 20:
                    Vol = fig_voltage.marks[0]
                    C_Na = fig_C_Na.marks[0]
                    C_K = fig_C_K.marks[0]
                case 40:
                    Vol = fig_voltage.marks[1]
                    C_Na = fig_C_Na.marks[1]
                    C_K = fig_C_K.marks[1]
                case 60:
                    Vol = fig_voltage.marks[2]
                    C_Na = fig_C_Na.marks[2]
                    C_K = fig_C_K.marks[2]
                case _:
                    Vol = fig_voltage.marks[3]
                    C_Na = fig_C_Na.marks[3]
                    C_K = fig_C_K.marks[3]
                        
            for i in range(len(V)):
                time.sleep(0.025)
                
                with Vol.hold_sync(), C_Na.hold_sync(), C_K.hold_sync():
                    Vol.x = X.t
                    Vol.y = V[:i]
                    C_Na.x = X.t
                    C_Na.y = na_conductance[:i]
                    C_K.x = X.t
                    C_K.y = k_conductance[:i]
                    

        b20.on_click(functools.partial(play, Vc=20))
        b40.on_click(functools.partial(play, Vc=40))
        b60.on_click(functools.partial(play, Vc=60))
                    
        display(VBox([b20, b40, b60, fig_C_Na, fig_C_K, fig_voltage], layout=Layout(align_items='center', width='500px')))

    def plot_AP(self):
        """
        calculate and plot
        """
        interval = (0,50)

        # labels: MathJax not supported with bqplot
        fig_conductance = self.init_figure('700px', '200px', 'Conductance (K)', interval, (-1, 50), 2, True)
        fig_voltage = self.init_figure('700px', '400px', 'Membrane voltage', interval, (-100, 100), 1, True)
        
        b20 = Button(description='-50 mV')
        b40 = Button(description='10 mV')
        b60 = Button(description='15 mV')
                
        def play(b, Vc):
            self.reset_parms()
            X = self.simulation(interval, False, [(5, Vc)])
            #print(X.y)
            V, m, h, n = X.y
            depol = False
            
            na_conductance = self.C_Na(m, h)
            k_conductance = self.C_K(n)
            
            # index for label timesteps: 0=depolarisation, 1=repolarisation, 2=hyperpolarisation
            j = 0
            
            # positioning offset of the labels
            offset = np.array([[-1, 10],
                               [0, 10],
                               [1, -15]])
            
            # reset (hide) AP label marks
            # TODO mark counter
            for mark in fig_voltage.marks[3:] + fig_conductance.marks[2:]:
                mark.x = [-50]

            Vol = fig_voltage.marks[0]
            C1 = fig_conductance.marks[0]
            C2 = fig_conductance.marks[1]
            
            if np.max(V) > -55:
                depol = True
                idx_repol = np.argmax(V)
                timesteps = [np.argmax(V > -55), idx_repol, idx_repol + np.argmax(V[idx_repol:] < self.Em)]

            for i in range(len(X.t)):
                time.sleep(0.025)
                
                if depol:
                    if i == timesteps[j] + 2:
                        fig_voltage.marks[2*(j+1) + 1].x = [X.t[timesteps[j]], X.t[timesteps[j]]]
                        fig_conductance.marks[2*(j+2)].x = [X.t[timesteps[j]], X.t[timesteps[j]]]
                        
                        fig_voltage.marks[2*(j+1) + 2].x = [X.t[timesteps[j]] + offset[j][0]]
                        fig_voltage.marks[2*(j+1) + 2].y = [V[timesteps[j]] + offset[j][1]]
                        j += 1
                        if j > 2:
                            depol = False
                        time.sleep(1)
                
                with Vol.hold_sync(), C1.hold_sync(), C2.hold_sync():
                    C1.x = X.t
                    C1.y = na_conductance[:i]
                    C2.x = X.t
                    C2.y = k_conductance[:i]
                    Vol.x = X.t
                    Vol.y = V[:i]

        b20.on_click(functools.partial(play, Vc=-100))
        b40.on_click(functools.partial(play, Vc=10))
        b60.on_click(functools.partial(play, Vc=15))
                    
        display(VBox([b20, b40, b60, fig_voltage, fig_conductance], layout=Layout(align_items='center', width='800px')))

    def plot_AP_TTX_TEA(self):
        """
        calculate and plot
        """
        interval = (0,50)
        
        dropdown = Dropdown(description="Select:",
                            options=['Tetrodotoxin (TTX)', 'Tetraethylammonium (TEA)'])

        # labels: MathJax not supported with bqplot
        fig_conductance = self.init_figure('700px', '200px', 'Conductance (K)', interval, (-1, 50), 2)
        fig_voltage = self.init_figure('700px', '400px', 'Membrane voltage', interval, (-100, 100), 1)
        
        b20 = Button(description='-50 mV')
        b40 = Button(description='10 mV')
        b60 = Button(description='15 mV')
                
        def play(b, Vc):
            self.set_parms([40.0, 36.0, 0.3], np.array([-115.0, 12.0, 0.0]))
            X = self.simulation(interval, False, [(5, Vc)],
                                dropdown.value == 'Tetrodotoxin (TTX)',
                                dropdown.value == 'Tetraethylammonium (TEA)')

            V, m, h, n = X.y
            na_conductance = self.C_Na(m, h)
            k_conductance = self.C_K(n)

            Vol = fig_voltage.marks[0]
            C1 = fig_conductance.marks[0]
            C2 = fig_conductance.marks[1]

            for i in range(len(X.t)):
                time.sleep(0.025)
                
                with Vol.hold_sync(), C1.hold_sync(), C2.hold_sync():
                    C1.x = X.t
                    C1.y = na_conductance[:i]
                    C2.x = X.t
                    C2.y = k_conductance[:i]
                    Vol.x = X.t
                    Vol.y = V[:i]

        b20.on_click(functools.partial(play, Vc=-100))
        b40.on_click(functools.partial(play, Vc=10))
        b60.on_click(functools.partial(play, Vc=55))
                    
        display(VBox([dropdown, b60, fig_voltage, fig_conductance], layout=Layout(align_items='center', width='800px')))
        

    def plot_refractory(self):
        """
        calculate and plot
        """
        interval = (0,50)

        fig_voltage = self.init_figure('700px', '400px', 'Membrane voltage', interval, (-100, 100), 1)
        
        # create a fast interval selector by passing in the X scale and the line mark on which the selector operates
        intsel = BrushIntervalSelector(marks=fig_voltage.marks, scale=fig_voltage.marks[0].scales["x"])
        fig_voltage.interaction = intsel  # set the interval selector on the figure

        
        b20 = Button(description='-50 mV')
        b40 = Button(description='10 mV')
        b60 = Button(description='15 mV')
                
        def play(b, Vc):
            self.reset_parms()
            X = self.simulation(interval, False, [(intsel.selected[0], Vc), (intsel.selected[1], Vc)])

            V, _, _, _ = X.y

            Vol = fig_voltage.marks[0]

            for i in range(len(X.t)):
                time.sleep(0.025)
                
                with Vol.hold_sync():
                    Vol.x = X.t
                    Vol.y = V[:i]

        b20.on_click(functools.partial(play, Vc=-100))
        b40.on_click(functools.partial(play, Vc=10))
        b60.on_click(functools.partial(play, Vc=55))
        
        display(VBox([b60, fig_voltage], layout=Layout(align_items='center', width='800px')))
        
        
    def plot_explorer(self):
        """
        calculate and plot
        """
        interval = (0,50)
        
        fig_conductance = self.init_figure('700px', '200px', 'Conductance (K)', interval, (-1, 50), 4)
        fig_voltage = self.init_figure('700px', '400px', 'Membrane voltage', interval, (-100, 100), 2)
        
        # create a fast interval selector by passing in the X scale and the line mark on which the selector operates
        intsel = BrushIntervalSelector(marks=fig_voltage.marks, scale=fig_voltage.marks[0].scales["x"])
        fig_voltage.interaction = intsel  # set the interval selector on the figure

        gNa = BoundedIntText(value=self.gNa, min=self.gNa - 50, max=self.gNa + 50, step=1, description='gNa:', disabled=False)
        gK = BoundedIntText(value=self.gK, min=self.gK - 50, max=self.gK + 50, step=1, description='gK:', disabled=False)
        ENa = BoundedIntText(value=self._V(self.ENa), min=self._V(self.ENa) - 50, max=self._V(self.ENa) + 50, step=1, description='ENa:', disabled=False)
        EK = BoundedIntText(value=self._V(self.EK), min=self._V(self.EK) - 50, max=self._V(self.EK) + 50, step=1, description='EK:', disabled=False)

        # draw initial lines
        Vc = 50
        X = self.simulation(interval, False, [(5, Vc)])

        V, m, h, n = X.y
        na_conductance = self.C_Na(m, h)
        k_conductance = self.C_K(n)

        Vol = fig_voltage.marks[0]
        C1 = fig_conductance.marks[0]
        C2 = fig_conductance.marks[1]
        
        Vol.line_style='dashed'
        C1.line_style='dashed'
        C2.line_style='dashed'
        Vol.stroke_width = .5

        with Vol.hold_sync(), C1.hold_sync(), C2.hold_sync():
            C1.x = X.t
            C1.y = na_conductance
            C2.x = X.t
            C2.y = k_conductance
            Vol.x = X.t
            Vol.y = V

        def on_change(change):
            Vc=50
            
            self.gNa = gNa.value            
            self.gK = gK.value            
            self.ENa = self._V(ENa.value)
            self.EK = self._V(EK.value)
            
            X = self.simulation(interval, False, [(5, Vc)])

            V, m, h, n = X.y
            na_conductance = self.C_Na(m, h)
            k_conductance = self.C_K(n)

            Vol = fig_voltage.marks[1]
            C1 = fig_conductance.marks[2]
            C2 = fig_conductance.marks[3]

            with Vol.hold_sync(), C1.hold_sync(), C2.hold_sync():
                C1.x = X.t
                C1.y = na_conductance
                C2.x = X.t
                C2.y = k_conductance
                Vol.x = X.t
                Vol.y = V
        
        gNa.observe(on_change, names='value')
        gK.observe(on_change, names='value')
        ENa.observe(on_change, names='value')
        EK.observe(on_change, names='value')
        
        display(VBox([gNa, gK, ENa, EK, fig_voltage, fig_conductance], layout=Layout(align_items='center', width='800px')))
        
        
