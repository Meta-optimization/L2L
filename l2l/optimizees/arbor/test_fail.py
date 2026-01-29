from interface import load_ref, load_params, ArborParRunner
from prior import labels
import numpy as np

def get_features(ts, um):
    # resting potential
    u_rest = um[30_000] # at t=150 ms

    # spike identification
    spike_idx = []
    for i, u in enumerate(um[1:], 1):
        if u > -20 and um[i-1] <= -20:
            spike_idx.append(i)
    spike_idx = np.array(spike_idx)
    if len(spike_idx) > 0:
        spike_ts = ts[spike_idx]
    else:
        spike_ts = []

    # initial spike time
    if len(spike_ts) > 0:
        first_spike_t = spike_ts[0]*1e-3
    else:
        first_spike_t = 0

    # spike frequency/average firing rate
    avg_rate = len(spike_ts)/(ts[-1]-200)*1e3

    # average inter-spike interval and trough height
    if len(spike_ts) > 1:
        avg_isi = np.mean(spike_ts[1:]-spike_ts[:-1])*1e-3
        through_height = np.mean(um[(spike_idx[1:]+spike_idx[:-1])//2])
    else:
        avg_isi = 0
        through_height = 0

    # action potential height and width
    if len(spike_ts) > 0:
        ap_height = np.mean([np.max(um[i:i+1000]) for i in spike_idx]) # i:i+1000 means next 5 ms

        val = (u_rest+ap_height)//2
        start = 0
        tmp = []
        for i, u in enumerate(um[1:], 1):
            if u > val and um[i-1] <= val:
                start = ts[i]
                continue
            if um[i-1] > val and u <= val:
                tmp.append(ts[i]-start)
        ap_width = np.mean(tmp)
    else:
        ap_height = 0
        ap_width = 0

    return (u_rest, first_spike_t, avg_rate, avg_isi, through_height, ap_height, ap_width)

ref = load_ref('/home/todt/Dokumente/L2L/l2l/optimizees/arbor/ref.csv') # Vergleichskurve ("observation")
par = load_params('/home/todt/Dokumente/L2L/l2l/optimizees/arbor/fit.json')
opt = ArborParRunner()

tensor = float
individual =  {
    'soma_gbar_NaV': 0.0623377189040184,
    'axon_gbar_NaV': 0.020993605256080627,
    'dend_gbar_NaV': 0.06141170859336853,
    'apic_gbar_NaV': 0.00021103672042954713,
    'soma_gbar_Kv3_1': 0.1536858230829239,
    'axon_gbar_Kv3_1': 0.5232238173484802,
    'dend_gbar_Kv3_1': 0.15744256973266602,
    'apic_gbar_Kv3_1': 0.8241668939590454,
    'soma_gbar_Ca_HVA': 8.046084258239716e-05,
    'soma_gbar_Ca_LVA': 0.002508891746401787,
    'axon_gbar_Ca_HVA': 3.8060647966631223e-06,
    'axon_gbar_Ca_LVA': 0.0056434012949466705
}

# replace parameters
for label in labels:
    par[label] = float(individual[label])

ts, um = opt.run(par)

# summary statistics
obs = get_features(ts, um)

print(obs)