import arbor as A
from arbor import units as U
from pathlib import Path
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

here = Path(__file__).parent


def load_ref(fn="ref.csv"):
    res = pd.read_csv(fn, usecols=[1, 2])
    res['U/mV'] *= 1000 # V -> mV
    res['U/mV'] -= 14   # add junction shift from fit/fitting/junction_potential
    return res


def load_params(fn="fit.json"):
    with open(here / fn) as fd:
        fit = json.load(fd)

    res = {}
    # Get default parameters
    res["temp"] = float(fit["conditions"][0]["celsius"])
    res["Vm"] = float(fit["conditions"][0]["v_init"])
    res["Ra"] = float(fit["passive"][0]["Ra"])

    for block in fit["genome"]:
        region = block["section"]
        name = block["name"]
        value = float(block["value"])
        res[f"{region}_{name}"] = value

    # Get ion reversal potentials
    for kv in fit["conditions"][0]["erev"]:
        region = kv["section"]
        for k, v in kv.items():
            if k == "section":
                continue
            ion = k[1:]
            res[f"{region}_{ion}_erev"] = float(v)
    return res


class ArborRunner:
    def __init__(self):
        self.regions = ["soma", "dend", "apic", "axon"]
        self.ions = ["na", "ca", "k"]
        # stop time of simulation
        # NOTE: This should be configurable
        self.T = 1400
        self.dt = 0.0025
        self.morphology = here / "allen.swc"
        self.cvp = A.cv_policy_max_extent(20)  # um
        # extract mechanisms and their parameters
        self.mechs = []
        for k, v in A.default_catalogue().items():
            ps = list(v.parameters.keys()) + list(v.globals.keys())
            self.mechs.append((k, ps))
        for k, v in A.allen_catalogue().items():
            ps = list(v.parameters.keys()) + list(v.globals.keys())
            self.mechs.append((k, ps))

        # current clamps: where, start/ms, duration/ms, current/nA
        # NOTE: This should be configurable
        self.i_clamps = [("(location 0 0.5)", 200, 1000, 0.150)]

    def run(self, params):
        dec = A.decor()
        # Set default properties
        if "temp" in params:
            dec.set_property(tempK=params["temp"] * U.Celsius)
        if "Vm" in params:
            dec.set_property(Vm=params["Vm"] * U.mV)
        if "cm" in params:
            dec.set_property(cm=params["cm"] * U.uF / U.cm2)
        if "Ra" in params:
            dec.set_property(rL=params["Ra"] * U.Ohm * U.cm)

        # Override region parameters
        for region in self.regions:
            key = f"{region}_temp"
            if key in params:
                dec.paint(f'"{region}"', tempK=params[key] * U.Celsius)
            key = f"{region}_Vm"
            if key in params:
                dec.paint(f'"{region}"', Vm=params[key] * U.mV)
            key = f"{region}_cm"
            if key in params:
                dec.paint(
                    f'"{region}"',
                    cm=params[key] * U.uF / U.cm2,
                )
            key = f"{region}_Ra"
            if key in params:
                dec.paint(
                    f'"{region}"',
                    rL=params[key] * U.Ohm * U.cm,
                )

        for region in self.regions:
            for mech, keys in self.mechs:
                # de-compress flat dictionary so we can use it with `paint`
                ps = {}
                for key in keys:
                    k = f"{region}_{key}_{mech}"
                    if k in params:
                        ps[key] = params[k]
                if not ps:
                    continue
                # special treatment of pas
                if mech == "pas" and "e" in ps:
                    mech = f"{mech}/e={ps['e']}"
                    ps.pop("e")
                dec.paint(f'"{region}"', A.density(A.mechanism(mech, ps)))

        # Add current clamps
        for ix, (loc, t0, dt, ic) in enumerate(self.i_clamps):
            dec.place(loc, A.iclamp(t0 * U.ms, dt * U.ms, ic * U.nA), f"ic-{ix}")

        # Set ion reversal potentials
        for region in self.regions:
            for ion in self.ions:
                key = f"{region}_{ion}_erev"
                if key in params:
                    dec.paint(f'"{region}"', ion=ion, rev_pot=params[key] * U.mV)

        # Set calcium Nernst equation
        # NOTE: This should be configurable, but we must use a kludge to emulate Neuron's
        #       weird Nernst rule
        dec.set_ion("ca", int_con=5e-5 * U.mM, ext_con=2.0 * U.mM, method="nernst/x=ca")

        # load morphology
        mrf = A.load_swc_neuron(self.morphology)

        # Create cell
        cell = A.cable_cell(mrf.morphology, dec, mrf.labels, self.cvp)

        sim = A.single_cell_model(cell)
        sim.properties.catalogue.extend(A.allen_catalogue(), "")

        sim.probe("voltage", "(location 0 0.5)", "Um", frequency=200 * U.kHz)

        sim.run(tfinal=self.T * U.ms, dt=self.dt * U.ms)

        return np.array(sim.traces[0].time), np.array(sim.traces[0].value)

class ArborParRunner:
    def __init__(self):
        self.regions = ["soma", "dend", "apic", "axon"]
        self.ions = ["na", "ca", "k"]
        # stop time of simulation
        # NOTE: This should be configurable
        self.T = 1400
        self.dt = 0.0025
        self.morphology = here / "allen.swc"
        self.cvp = A.cv_policy_max_extent(20)  # um
        # extract mechanisms and their parameters
        self.mechs = []
        for k, v in A.default_catalogue().items():
            ps = list(v.parameters.keys()) + list(v.globals.keys())
            self.mechs.append((k, ps))
        for k, v in A.allen_catalogue().items():
            ps = list(v.parameters.keys()) + list(v.globals.keys())
            self.mechs.append((k, ps))

        # current clamps: where, start/ms, duration/ms, current/nA
        # NOTE: This should be configurable
        self.i_clamps = [("(location 0 0.5)", 200, 1000, 0.150)]

    def run(self, params):
        dec = A.decor()
        # Set default properties
        if "temp" in params:
            dec.set_property(tempK=params["temp"] * U.Celsius)
        if "Vm" in params:
            dec.set_property(Vm=params["Vm"] * U.mV)
        if "cm" in params:
            dec.set_property(cm=params["cm"] * U.uF / U.cm2)
        if "Ra" in params:
            dec.set_property(rL=params["Ra"] * U.Ohm * U.cm)

        # Override region parameters
        for region in self.regions:
            key = f"{region}_temp"
            if key in params:
                dec.paint(f'"{region}"', tempK=params[key] * U.Celsius)
            key = f"{region}_Vm"
            if key in params:
                dec.paint(f'"{region}"', Vm=params[key] * U.mV)
            key = f"{region}_cm"
            if key in params:
                dec.paint(
                    f'"{region}"',
                    cm=params[key] * U.uF / U.cm2,
                )
            key = f"{region}_Ra"
            if key in params:
                dec.paint(
                    f'"{region}"',
                    rL=params[key] * U.Ohm * U.cm,
                )

        for region in self.regions:
            for mech, keys in self.mechs:
                # de-compress flat dictionary so we can use it with `paint`
                ps = {}
                for key in keys:
                    k = f"{region}_{key}_{mech}"
                    if k in params:
                        ps[key] = params[k]
                if not ps:
                    continue
                # special treatment of pas
                if mech == "pas" and "e" in ps:
                    mech = f"{mech}/e={ps['e']}"
                    ps.pop("e")
                dec.paint(f'"{region}"', A.density(A.mechanism(mech, ps)))

        # Set ion reversal potentials
        for region in self.regions:
            for ion in self.ions:
                key = f"{region}_{ion}_erev"
                if key in params:
                    dec.paint(f'"{region}"', ion=ion, rev_pot=params[key] * U.mV)

        # Set calcium Nernst equation
        # NOTE: This should be configurable, but we must use a kludge to emulate Neuron's
        #       weird Nernst rule
        dec.set_ion("ca", int_con=5e-5 * U.mM, ext_con=2.0 * U.mM, method="nernst/x=ca")

        # load morphology
        mrf = A.load_swc_neuron(self.morphology)

        ts = None
        ums = []
        for ix, (loc, t0, dt, ic) in enumerate(self.i_clamps):
            tmp = A.decor(dec)
            tmp.place(loc, A.iclamp(t0 * U.ms, dt * U.ms, ic * U.nA), f"ic-{ix}")
            # Create cell
            cell = A.cable_cell(mrf.morphology, tmp, mrf.labels, self.cvp)
            sim = A.single_cell_model(cell)
            sim.properties.catalogue.extend(A.allen_catalogue(), "")
            sim.probe("voltage", "(location 0 0.5)", "Um", frequency=200 * U.kHz)
            sim.run(tfinal=self.T * U.ms, dt=self.dt * U.ms)
            ts = np.array(sim.traces[0].time)
            ums.append(np.array(sim.traces[0].value))
        return ts, ums[0]

if __name__ == "__main__":
    ref = load_ref()
    par = load_params()
    opt = ArborRunner()
    ts, um = opt.run(par)
    fg, ax = plt.subplots()
    ax.plot(ts, um)
    ax.plot(ref['t/ms'], ref['U/mV'])
    fg.savefig("plot.pdf")