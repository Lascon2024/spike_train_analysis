import numpy as np
import quantities as pq
import neo
import elephant.spike_train_generation as stg
import matplotlib.pyplot as plt


def generate_spike_trains_with_coinc(lambda_b, lambda_c, trial_duration, coinc_duration, num_trials,
                                     num_units=2, unit_ids_sync=(1,2), RateJitter=0*pq.Hz):
    """
    generate stationary poisson spiketrains with injected coincidences
    """
    if not isinstance(unit_ids_sync[0], (list, tuple, np.ndarray)):
        unit_ids_sync = [unit_ids_sync]
    
    sync_count = np.zeros(num_units, dtype=int)
    for unit_ids in unit_ids_sync:
        for unit_id in unit_ids:
            sync_count[unit_id - 1] += 1

    t_coinc_start = (trial_duration - coinc_duration) / 2.
    t_coinc_stop = t_coinc_start + coinc_duration

    rate_jitters = (np.random.rand(num_trials) - 0.5) * RateJitter
    spike_trains = []
    for i_trial in range(num_trials):
        rate = lambda_b + rate_jitters[i_trial]

        # spiketrains of non-coincident spikes
        sp_pre_coinc = [stg.homogeneous_poisson_process(rate, 0.*pq.ms, t_coinc_start) for _ in range(num_units)]
        sp_coinc = [stg.homogeneous_poisson_process(rate - lambda_c*sync_count[i], t_coinc_start, t_coinc_stop) for i in range(num_units)]
        sp_post_coinc = [stg.homogeneous_poisson_process(rate, t_coinc_stop, trial_duration) for _ in range(num_units)]

        # spiketrains of coincident spikes (one spiketrain per synchronous subset of units)
        coinc = [stg.homogeneous_poisson_process(lambda_c, t_coinc_start, t_coinc_stop) for _ in unit_ids_sync]

        sts = []
        for i_unit in range(num_units):
            # collect all spike times of a unit
            spike_times = [sp_pre_coinc[i_unit].times.rescale('ms').magnitude,
                           sp_coinc[i_unit].times.rescale('ms').magnitude,
                           sp_post_coinc[i_unit].times.rescale('ms').magnitude]
            for i, unit_ids in enumerate(unit_ids_sync):
                if i_unit + 1 in unit_ids:
                    spike_times.append(coinc[i].times.rescale('ms').magnitude)

            # concatenate the collected spike times and sort them
            spike_times = np.sort(np.concatenate(spike_times))

            sts.append(neo.SpikeTrain(spike_times*pq.ms, t_start=0.*pq.ms, t_stop=trial_duration))
        
        spike_trains.append(sts)            

    return spike_trains


def generate_spike_trains_with_osc_coinc(num_trials, num_units, trial_duration, freq_coinc, amp_coinc, offset_coinc,
                                         freq_bg, amp_bg, offset_bg, RateJitter=10*pq.Hz):
    """
    generate non-stationary poisson spiketrains with oscillatory rate modulation and injected coincidences
    """
    dt = 1 * pq.ms
    times = np.arange(0, trial_duration.rescale('s').magnitude, dt.rescale('s').magnitude)
    pi2 = np.pi * 2

    # modulatory coincidence rate
    phases_coinc = pi2 * freq_coinc.rescale('Hz').magnitude * times
    rate_coinc = (offset_coinc + amp_coinc * np.sin(phases_coinc)).rescale('Hz').magnitude
    rate_coinc[rate_coinc < 0] = 0

    # background rate
    phases_bg = pi2 * freq_bg.rescale('Hz').magnitude * times
    rate_bg = (offset_bg + amp_bg * np.sin(phases_bg)).rescale('Hz').magnitude
    rate_bg[rate_bg < 0] = 0

    # inhomogenious rate across trials
    rate_jitters = (np.random.rand(num_trials) - 0.5) * RateJitter
    spiketrain = []
    for i in range(num_trials):
        rate_signal_bg = neo.AnalogSignal(rate_bg + rate_jitters[i].magnitude, sampling_period=dt, units=pq.Hz, t_start=0*pq.ms)
        rate_signal_coinc = neo.AnalogSignal(rate_coinc, sampling_period=dt, units=pq.Hz, t_start=0*pq.ms)
        sts_bg = [stg.inhomogeneous_poisson_process(rate_signal_bg) for _ in range(num_units)]
        # inserting coincidences
        sts_coinc = stg.inhomogeneous_poisson_process(rate_signal_coinc)
        sts_bg_coinc = []
        for st_bg in sts_bg:
            spike_times = np.sort(np.append(st_bg.times.magnitude, sts_coinc.times.magnitude))
            st_bg_coinc = neo.SpikeTrain(spike_times, units=st_bg.units, t_start=st_bg.t_start, t_stop=st_bg.t_stop)
            sts_bg_coinc.append(st_bg_coinc)
        spiketrain.append(sts_bg_coinc)
    return {'st':spiketrain, 'background_rate':rate_bg, 'coinc_rate':rate_coinc}
