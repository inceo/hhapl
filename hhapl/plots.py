import numpy as np
import time

# ipywidgets as interactive Widgets for the Jupyter notebooks (and Voila)
from ipywidgets import Layout, VBox, HBox, Button, Dropdown, \
    BoundedIntText, IntSlider, Output

from IPython.display import display, clear_output

# bqplot as plotting library for Jupyter notebooks
import bqplot.pyplot as plt
from bqplot.interacts import BrushIntervalSelector

# project classes
from hhapl import HodgkinHuxley

# Tweak for working with ipywidgets
# https://github.com/jupyter-widgets/ipywidgets/issues/2103
import functools


class Plots(HodgkinHuxley.HodgkinHuxley):
    """
    Plot class for the Hodgkin-Huxley action potential lab

    It generates all plots displayed in the application, initilizes bqplot
    figures and interactive Widgets and interacts with user input and manages
    the animations.
    """

    def __init__(self):
        """
        Initilizing the plot class and inheriting Hodgkin-Huxley model class
        """
        super().__init__()

    def init_figure(self, width, height, label, xlim, ylim,
                    num_lines, ap_labels=False):
        """
        Initilize figure

        bqplot allows only to change (adding elements such as marks) the
        current figure, but we are mostly dealing with multiple-figure-plots
        and since subplots are not available in bqplot, we pre-register or
        initilize all elements (marks). We do this to have a consistent order
        to later animate or manipulate specific marks.

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
        colors = [['blue'], ['green'], ['red'], ['orange']]

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
        fig.fig_margin = {'top': 10, 'bottom': 35, 'left': 50, 'right': 50}

        fig.legend = False

        return fig

    def plot_Na(self):
        """
        Plot for sodium ion channel. Demonstrates the inactivation.
        Initilizes Widgets and plots and runs animation.

        This function is to be called from a Jupyter notebook.
        """

        interval = (0, 20)
        button_config = [20, 40, 60, 0]

        fig_voltage = self.init_figure('400px', '200px', 'Clamped voltage',
                                       interval, (-1, 100), 4)
        fig_conductance = self.init_figure('400px', '200px', 'Conductance',
                                           interval, (-1, 40), 4)

        buttons = [Button(description='mV') if v == 0
                   else Button(description=str(v) + ' mV')
                   for v in button_config]

        slider = IntSlider(value=50, min=1, max=100, step=1,
                           continuous_update=True,
                           readout=True)

        out = Output()

        def play(b, bidx):
            """
            Mark animation function

            Parameters
            ----------
            b : ipywidget Button object
                needs to be passed as firsnt parameter
            bidx : int
                Button index which is used to get the voltage
            """

            # reset parms because parameters can be manipulated during runtime
            # and here we want the standard parameters
            self.reset_parms()
            with out:
                clear_output()

            if button_config[bidx] == 0:
                vol = slider.value
            else:
                vol = button_config[bidx]

            # call simulation
            res = self.simulation(interval, True, [5, 15, vol])

            # check for excetions
            if isinstance(res, str):
                with out:
                    print(res)
            else:
                V = vol * np.logical_and(res.t > 5,
                                         res.t < 15)

                # for sodium,only m and h (activation and inactivation)
                m, h, _ = res.y

                # calculate the sodium conductance beforehand (performance)
                na_conductance = self.cond_na(m, h)

                # get correct line corresponding to pressed Button
                line_voltage = fig_voltage.marks[bidx]
                line_conductancu = fig_conductance.marks[bidx]

                # animation loop
                for i in range(len(V)):
                    # animation speed
                    time.sleep(0.025)

                    with line_voltage.hold_sync(), \
                            line_conductancu.hold_sync():
                        line_voltage.x = res.t
                        line_voltage.y = V[:i]
                        line_conductancu.x = res.t
                        line_conductancu.y = na_conductance[:i]

        for idx, b in enumerate(buttons):
            b.on_click(functools.partial(play, bidx=idx))

        display(VBox([*buttons[:3],
                      HBox([slider, buttons[3]]),
                      fig_conductance, fig_voltage, out],
                     layout=Layout(align_items='center', width='500px')))

    def plot_K(self):
        """
        Plot for potassium and sodium ion channel. Demonstrates the
        voltage-dependance of the K+ channel.

        Initilizes Widgets and plots and runs animation.

        This function is to be called from a Jupyter notebook.
        """

        interval = (0, 20)
        button_config = [20, 40, 60, 0]

        fig_voltage = self.init_figure('400px', '200px', 'Clamped voltage',
                                       interval, (-1, 100), 4)
        fig_conductance_na = self.init_figure('400px', '200px', 'Conductance',
                                              interval, (-1, 50), 4)
        fig_conductance_k = self.init_figure('400px', '200px', 'Conductance',
                                             interval, (-1, 50), 4)

        buttons = [Button(description='mV') if v == 0
                   else Button(description=str(v) + ' mV')
                   for v in button_config]

        slider = IntSlider(value=50, min=1, max=100, step=1,
                           continuous_update=True,
                           readout=True,
                           width=50)

        out = Output()

        def play(b, bidx):
            """
            Mark animation function

            Parameters
            ----------
            b : ipywidget Button object
                needs to be passed as firsnt parameter
            bidx : int
                Button index which is used to get the voltage
            """

            # reset parms because parameters can be manipulated during runtime
            # and here we want the standard parameters
            self.reset_parms()
            with out:
                clear_output()

            if button_config[bidx] == 0:
                vol = slider.value
            else:
                vol = button_config[bidx]

            # call simulation
            res = self.simulation(interval, True, [5, 15, vol])

            # check for excetions
            if isinstance(res, str):
                with out:
                    print(res)
            else:
                V = vol * np.logical_and(res.t > 5,
                                         res.t < 15)

                # for sodium, only m and h (activation and inactivation)
                # for potassium, n (activation)
                m, h, n = res.y

                # calculate the sodium conductance beforehand (performance)
                na_conductance = self.cond_na(m, h)
                k_conductance = self.cond_k(n)

                # get correct line corresponding to pressed Button
                line_voltage = fig_voltage.marks[bidx]
                line_conductance_na = fig_conductance_na.marks[bidx]
                line_conductance_k = fig_conductance_k.marks[bidx]

                # animation loop
                for i in range(len(V)):
                    # animation speed
                    time.sleep(0.025)

                    with line_voltage.hold_sync(), \
                            line_conductance_k.hold_sync(), \
                            line_conductance_na.hold_sync():
                        line_voltage.x = res.t
                        line_voltage.y = V[:i]
                        line_conductance_na.x = res.t
                        line_conductance_na.y = na_conductance[:i]
                        line_conductance_k.x = res.t
                        line_conductance_k.y = k_conductance[:i]

        for idx, b in enumerate(buttons):
            b.on_click(functools.partial(play, bidx=idx))

        display(VBox([*buttons[:3],
                      HBox([slider, buttons[3]]),
                      fig_conductance_na, fig_conductance_k, fig_voltage, out],
                     layout=Layout(align_items='center', width='500px')))

    def plot_AP(self):
        """
        Plot for action potential.

        Initilizes Widgets and plots and runs animation.

        This function is to be called from a Jupyter notebook.
        """
        interval = (0, 30)
        button_config = [5, 15, 0]

        fig_conductance = self.init_figure('700px', '200px',
                                           'Conductance (K)',
                                           interval, (-1, 50), 2, True)

        fig_voltage = self.init_figure('700px', '400px',
                                       'Membrane voltage',
                                       interval, (-120, 80), 1, True)

        buttons = [Button(description='mV') if v == 0
                   else Button(description=str(v) + ' mV')
                   for v in button_config]

        slider = IntSlider(value=50, min=-80, max=100, step=1,
                           continuous_update=True,
                           readout=True)

        out = Output()

        def play(b, bidx):
            """
            Mark animation function

            Parameters
            ----------
            b : ipywidget Button object
                needs to be passed as firsnt parameter
            bidx : int
                Button index which is used to get the voltage
            """

            # reset parms because parameters can be manipulated during runtime
            # and here we want the standard parameters
            self.reset_parms()
            with out:
                clear_output()

            if button_config[bidx] == 0:
                vol = slider.value
            else:
                vol = button_config[bidx]

            # call simulation
            res = self.simulation(interval, False, [(5, 6, vol)])

            # check for excetions
            if isinstance(res, str):
                with out:
                    print(res)
            else:
                V, m, h, n = res.y
                depol = False

                na_conductance = self.cond_na(m, h)
                k_conductance = self.cond_k(n)

                # index for label timesteps:
                # 0=depolarisation, 1=repolarisation, 2=hyperpolarisation
                j = 0

                # positioning offset of the labels
                offset = np.array([[-1, 10],
                                   [0, 10],
                                   [1, -15]])

                # reset (hide) AP label marks
                for mark in fig_voltage.marks[3:] + fig_conductance.marks[2:]:
                    mark.x = [-50]

                line_voltage = fig_voltage.marks[0]
                line_conductance_na = fig_conductance.marks[0]
                line_conductance_k = fig_conductance.marks[1]

                # check for depolarisation
                if np.max(V) > -55:
                    depol = True
                    idx_repol = np.argmax(V)
                    timesteps = [np.argmax(V > -55), idx_repol,
                                 idx_repol + np.argmax(V[idx_repol:] < self.e_m)]

                # animation loop
                for i in range(len(res.t)):
                    # animation speed
                    time.sleep(0.025)

                    # display the lines and label for the stages of AP
                    if depol:
                        if i == timesteps[j] + 2:
                            fig_voltage.marks[2*(j+1) + 1].x = \
                                [res.t[timesteps[j]], res.t[timesteps[j]]]
                            fig_conductance.marks[2*(j+2)].x = \
                                [res.t[timesteps[j]], res.t[timesteps[j]]]

                            fig_voltage.marks[2*(j+1)+2].x = \
                                [res.t[timesteps[j]] + offset[j][0]]
                            fig_voltage.marks[2*(j+1)+2].y = \
                                [V[timesteps[j]] + offset[j][1]]

                            j += 1
                            if j > 2:
                                depol = False
                            time.sleep(1)

                    with line_voltage.hold_sync(), \
                            line_conductance_na.hold_sync(), \
                            line_conductance_k.hold_sync():
                        line_conductance_na.x = res.t
                        line_conductance_na.y = na_conductance[:i]
                        line_conductance_k.x = res.t
                        line_conductance_k.y = k_conductance[:i]
                        line_voltage.x = res.t
                        line_voltage.y = V[:i]

        for idx, b in enumerate(buttons):
            b.on_click(functools.partial(play, bidx=idx))

        display(VBox([*buttons[:2],
                      HBox([slider, buttons[2]]),
                      fig_voltage, fig_conductance, out],
                     layout=Layout(align_items='center', width='800px')))

    def plot_AP_TTX_TEA(self):
        """
        Plot for showing the influence of Tetrodotoxin and Tetraethylammonium.

        Initilizes Widgets and plots and runs animation.

        This function is to be called from a Jupyter notebook.
        """
        interval = (0, 30)

        dropdown = Dropdown(description="Select:",
                            options=['Tetrodotoxin (TTX)',
                                     'Tetraethylammonium (TEA)'])

        fig_conductance = self.init_figure('700px', '200px', 'Conductance (K)',
                                           interval, (-1, 50), 2)
        fig_voltage = self.init_figure('700px', '400px', 'Membrane voltage',
                                       interval, (-100, 80), 1)

        button = Button(description='50 mV')

        out = Output()

        def play(b):
            """
            Mark animation function

            Parameters
            ----------
            b : ipywidget Button object
                needs to be passed as first parameter
            """
            # here we need to tweak the model because with one of the ion
            # channels, the numerical intergration is unstable
            self.set_parms([40.0, 36.0, 0.3], np.array([55.0, -77.0, -65.0]))
            with out:
                clear_output()
            res = self.simulation(interval, False, [(5, 6, 55)],
                                  dropdown.value == 'Tetrodotoxin (TTX)',
                                  dropdown.value == 'Tetraethylammonium (TEA)')

            # check for excetions
            if isinstance(res, str):
                with out:
                    print(res)
            else:
                V, m, h, n = res.y
                na_conductance = self.cond_na(m, h)
                k_conductance = self.cond_k(n)

                line_voltage = fig_voltage.marks[0]
                line_conductance_na = fig_conductance.marks[0]
                line_conductance_k = fig_conductance.marks[1]

                for i in range(len(res.t)):
                    time.sleep(0.025)

                    with line_voltage.hold_sync(), \
                            line_conductance_na.hold_sync(), \
                            line_conductance_k.hold_sync():
                        line_conductance_na.x = res.t
                        line_conductance_na.y = na_conductance[:i]
                        line_conductance_k.x = res.t
                        line_conductance_k.y = k_conductance[:i]
                        line_voltage.x = res.t
                        line_voltage.y = V[:i]

        button.on_click(play)

        display(VBox([dropdown, button, fig_voltage, fig_conductance],
                     layout=Layout(align_items='center', width='800px')))

    def plot_spike_train(self):
        """
        Plot for spike trains.

        Initilizes Widgets and plots and runs animation.

        This function is to be called from a Jupyter notebook.
        """
        interval = (0, 50)

        fig_voltage = self.init_figure('700px', '400px', 'Membrane voltage',
                                       interval, (-100, 100), 1)

        # create a fast interval selector by passing in the res scale and
        # the line mark on which the selector operates
        intsel = BrushIntervalSelector(marks=fig_voltage.marks,
                                       scale=fig_voltage.marks[0].scales["x"])

        # set the interval selector on the figure
        fig_voltage.interaction = intsel

        button = Button(description='mV')

        slider = IntSlider(value=50, min=1, max=100, step=1,
                           continuous_update=True,
                           readout=True)

        out = Output()

        def play(b):
            """
            Mark animation function

            Parameters
            ----------
            b : ipywidget Button object
                needs to be passed as first parameter
            """
            self.reset_parms()
            with out:
                clear_output()
            if intsel.selected is not None:
                vol = slider.value
                res = self.simulation(interval, False,
                                      [(intsel.selected[0],
                                        intsel.selected[1],
                                       vol)])

                if isinstance(res, str):
                    with out:
                        print(res)
                else:
                    V, _, _, _ = res.y

                    Vol = fig_voltage.marks[0]

                    for i in range(len(res.t)):
                        time.sleep(0.025)

                        with Vol.hold_sync():
                            Vol.x = res.t
                            Vol.y = V[:i]

        button.on_click(play)

        display(VBox([HBox([slider, button]), fig_voltage, out],
                layout=Layout(align_items='center', width='800px')))

    def plot_refractory(self):
        """
        Plot for refractory periodes.

        Initilizes Widgets and plots and runs animation.

        This function is to be called from a Jupyter notebook.
        """
        interval = (0, 50)

        fig_voltage = self.init_figure('700px', '400px', 'Membrane voltage',
                                       interval, (-100, 100), 1)

        # create a fast interval selector by passing in the res scale and
        # the line mark on which the selector operates
        intsel = BrushIntervalSelector(marks=fig_voltage.marks,
                                       scale=fig_voltage.marks[0].scales["x"])

        # set the interval selector on the figure
        fig_voltage.interaction = intsel

        button = Button(description='mV')

        slider = IntSlider(value=50, min=1, max=100, step=1,
                           continuous_update=True,
                           readout=True)

        out = Output()

        def play(b):
            """
            Mark animation function

            Parameters
            ----------
            b : ipywidget Button object
                needs to be passed as first parameter
            """
            self.reset_parms()
            with out:
                clear_output()
            if intsel.selected is not None:
                vol = slider.value
                res = self.simulation(interval, False,
                                      [(intsel.selected[0],
                                        intsel.selected[0] + 1,
                                       vol),
                                       (intsel.selected[1],
                                        intsel.selected[1] + 1,
                                       vol)])

                if isinstance(res, str):
                    with out:
                        print(res)
                else:
                    V, _, _, _ = res.y

                    Vol = fig_voltage.marks[0]

                    for i in range(len(res.t)):
                        time.sleep(0.025)

                        with Vol.hold_sync():
                            Vol.x = res.t
                            Vol.y = V[:i]

        button.on_click(play)

        display(VBox([HBox([slider, button]), fig_voltage, out],
                layout=Layout(align_items='center', width='800px')))

    def plot_explorer(self):
        """
        Plot for parameter influence.

        Initilizes Widgets and plots and runs animation.

        This function is to be called from a Jupyter notebook.
        """
        self.reset_parms()
        interval = (0, 25)

        fig_conductance = self.init_figure('700px', '200px', 'Conductance (K)',
                                           interval, (-1, 50), 4)
        fig_voltage = self.init_figure('700px', '400px', 'Membrane voltage',
                                       interval, (-140, 80), 2)

        g_na = BoundedIntText(value=self.g_na,
                              min=self.g_na - 30,
                              max=self.g_na + 30,
                              step=10,
                              description='Sodium conductance:',
                              disabled=False)

        g_k = BoundedIntText(value=self.g_k,
                             min=self.g_k - 15,
                             max=self.g_k + 15,
                             step=10,
                             description='Potassium conductance',
                             disabled=False)

        e_na = BoundedIntText(value=self.e_na,
                              min=self.e_na - 50,
                              max=self.e_na + 50,
                              step=10,
                              description='Sodium reversal potential',
                              disabled=False)

        e_k = BoundedIntText(value=self.e_k,
                             min=self.e_k - 50,
                             max=self.e_k + 50,
                             step=10,
                             description='Potassium reversal potential',
                             disabled=False)

        out = Output()

        # draw initial lines
        vol = 50
        res = self.simulation(interval, False, [(5, 6, vol)])

        V, m, h, n = res.y
        na_conductance = self.cond_na(m, h)
        k_conductance = self.cond_k(n)

        line_voltage = fig_voltage.marks[0]
        line_conductance_na = fig_conductance.marks[0]
        line_conductance_k = fig_conductance.marks[1]

        line_voltage.line_style = 'dashed'
        line_conductance_k.line_style = 'dashed'
        line_conductance_na.line_style = 'dashed'
        line_voltage.stroke_width = .5
        line_conductance_k.stroke_width = .5
        line_conductance_na.stroke_width = .5

        with line_voltage.hold_sync(), \
                line_conductance_na.hold_sync(), \
                line_conductance_k.hold_sync():
            line_conductance_na.x = res.t
            line_conductance_na.y = na_conductance
            line_conductance_k.x = res.t
            line_conductance_k.y = k_conductance
            line_voltage.x = res.t
            line_voltage.y = V

        def on_change(change):
            with out:
                clear_output()

            vol = 50

            self.g_na = g_na.value
            self.g_k = g_k.value
            self.e_na = e_na.value
            self.e_k = e_k.value

            res = self.simulation(interval, False, [(5, 6, vol)])

            if isinstance(res, str):
                with out:
                    print(res)
            else:
                V, m, h, n = res.y
                na_conductance = self.cond_na(m, h)
                k_conductance = self.cond_k(n)

                line_voltage = fig_voltage.marks[1]
                line_conductance_na = fig_conductance.marks[2]
                line_conductance_k = fig_conductance.marks[3]

                with line_voltage.hold_sync(), \
                        line_conductance_na.hold_sync(), \
                        line_conductance_k.hold_sync():
                    line_conductance_na.x = res.t
                    line_conductance_na.y = na_conductance
                    line_conductance_k.x = res.t
                    line_conductance_k.y = k_conductance
                    line_voltage.x = res.t
                    line_voltage.y = V

        g_na.observe(on_change, names='value')
        g_k.observe(on_change, names='value')
        e_na.observe(on_change, names='value')
        e_k.observe(on_change, names='value')

        display(VBox([g_na, g_k, e_na, e_k, fig_voltage, fig_conductance, out],
                     layout=Layout(align_items='center', width='800px')))
