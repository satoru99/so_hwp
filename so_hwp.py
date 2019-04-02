import numpy as np
import scipy.fftpack as fft
from spt3g import core
from spt3g.core import G3Units as u

class G3InputFrameList(object):
    ''' Start G3Pipeline from a list of frames '''
    def __init__(self, frames):
        self.frames = frames
        self.first = True

    def __call__(self, frame):
        if self.first:
            self.first = False
            out = self.frames
            if frame is not None:
                out = out + [frame]
            return out
        elif frame is not None:
            return [frame]
        else:
            return []

class G3OutputFrameList(object):
    ''' Collect output frames of G3Pipeline '''
    def __init__(self, frames=[]):
        self.frames = frames

    def __call__(self, frame):
        if frame.type != core.G3FrameType.EndProcessing:
            self.frames.append(frame)


class Sim_Obs_Setup(object):
    ''' Start G3Pipeline with an Observation frame '''
    def __init__(self,
                 obs_id=0, obs_start=0 * u.s, obs_end=10 * u.s,
                 **kwargs):
        kwargs['obs_id'] = obs_id
        kwargs['obs_start'] = obs_start
        kwargs['obs_end'] = obs_end
        self.info = kwargs
        self.is_first_frame = True

    def __call__(self, frame=None):
        if self.is_first_frame:
            self.is_first_frame = False
            obsframe = core.G3Frame(core.G3FrameType.Observation)
            for key in self.info:
                obsframe[key] = self.info[key]
            return [obsframe]
        else:
            return []

class Sim_Detector_Setup(object):
    ''' Add a Calibration frame which contains a list of detectors '''
    def __init__(self, detectors = ['det1', 'det2']):
        self.detectors = detectors

    def __call__(self, frame):
        if frame.type == core.G3FrameType.Observation:
            detframe = core.G3Frame(core.G3FrameType.Calibration)
            detectors = core.G3VectorString(self.detectors)
            detframe['det_list'] = detectors
            return [frame, detframe]
        else:
            return

class Sim_Scan_Prepare(object):
    ''' Add Scan frames with empty timestreams '''
    def __init__(self, sample_rate=400 * u.Hz, interval=60 * u.s):
        self.observation = None
        self.detectors = None
        self.interval = int(interval)
        self.sample_rate = sample_rate

    def __call__(self, frame):
        if frame.type == core.G3FrameType.Observation:
            self.observation = frame
            return
        if (frame.type == core.G3FrameType.Calibration and
            'det_list' in frame):
            self.detectors = frame
            return
        if frame.type == core.G3FrameType.EndProcessing:
            assert self.observation is not None
            assert self.detectors is not None
            output = []
            start = int(self.observation['obs_start'])
            end = int(self.observation['obs_end'])
            t0 = start
            while t0 < end:
                t1 = min(t0 + self.interval, end)
                n_sample = int((t1 - t0) * self.sample_rate)
                scan_frame = core.G3Frame(core.G3FrameType.Scan)
                scan_frame['start'] = core.G3Time(int(t0))
                scan_frame['stop'] = core.G3Time(int(t0 + n_sample / self.sample_rate))
                data = core.G3TimestreamMap()
                for n in self.detectors['det_list']:
                    data[n] = core.G3Timestream(np.zeros(n_sample))
                data.start = scan_frame['start']
                data.stop = core.G3Time(int(t0 + (n_sample - 1) / self.sample_rate))
                scan_frame['det_timestream'] = data
                output.append(scan_frame)
                if t0 + self.interval >= end:
                    break
                t0 = int(scan_frame['stop'])
            output.append(frame)
            return output


class Sim_HWP_Raw(object):
    ''' Append simulated raw data of the HWP encoder '''
    def __init__(self,
                 speed = 2.0 * u.Hz,
                 count_per_rev = 50000,
                 sample_rate = 20. * u.kHz,
                 clock = 16 * u.MHz,
                 ref_offset = 0. * u.deg,
                 initial_angle = 0. * u.deg):
        self.speed = speed
        self.count_per_rev = count_per_rev
        self.sample_rate = sample_rate
        self.clock = clock
        self.ref_offset = ref_offset
        self.initial_angle = initial_angle
        self.t0 = None

    def __call__(self, frame):
        if frame.type == core.G3FrameType.Observation:
            self.obs_start = int(frame['obs_start'])
            self.obs_end = int(frame['obs_end'])
            return
        if frame.type != core.G3FrameType.Scan:
            return
        assert 'det_timestream' in frame
        start = int(frame['start'])
        stop = int(frame['stop'])

        det_n_sample = frame['det_timestream'].n_samples
        det_start = int(frame['det_timestream'].start)
        det_stop = int(frame['det_timestream'].stop)
        true_angle0 = self.initial_angle + 360 * u.deg * ((
            (det_start - self.obs_start) * self.speed) % 1.)
        true_angle1 = true_angle0 + 360 * u.deg * (
            (det_stop - det_start) * self.speed)
        true_angle = np.linspace(true_angle0, true_angle1, det_n_sample)
        true_angle %= 360 * u.deg
        true_angle = core.G3Timestream(true_angle)
        true_angle.start = core.G3Time(det_start)
        true_angle.stop = core.G3Time(det_stop)
        frame['hwp_sim_true'] = true_angle

        if self.t0 is None:
            self.t0 = start
            initial_count = (self.count_per_rev *
                (self.initial_angle - self.ref_offset) / (360 * u.deg))
            self.e0 = (initial_count +
                       (start - self.obs_start) * self.speed
                       * self.count_per_rev)
            self.eref = self.count_per_rev * (
                np.floor(self.e0 / self.count_per_rev) + 1)
            self.eref -= self.e0 - (self.e0 % (1 << 16))
            self.e0 = self.e0 % (1 << 16)
            self.c0 = 0
            self.t1s = (int(self.t0) // int(u.s) + 1) * int(u.s)
        clocks = core.G3VectorInt()
        encoders = core.G3VectorInt()
        clock_at_ref = core.G3VectorInt()
        encoder_at_ref = core.G3VectorInt()
        time_at_1s = core.G3VectorTime()
        clock_at_1s = core.G3VectorInt()
        n_sample = int((stop - self.t0) * self.sample_rate)
        for i in range(n_sample):
            clocks.append(int(np.int32(self.c0)))
            encoders.append(int(self.e0))
            dt = int(1. / self.sample_rate)
            t1 = self.t0 + dt
            c1 = self.c0 + self.clock / self.sample_rate
            e1 = self.e0 + self.speed * self.count_per_rev / self.sample_rate
            if t1 > self.t1s:
                c1s = np.interp(self.t1s - self.t0, [0, dt], [self.c0, c1])
                c1s = int(np.int32(int(c1s) % (1 << 32)))
                clock_at_1s.append(c1s)
                time_at_1s.append(core.G3Time(self.t1s))
                self.t1s += int(u.s)
            if e1 > self.eref:
                cref = np.interp(self.eref, [self.e0, e1], [self.c0, c1])
                cref = int(np.int32(int(cref) % (1<<32)))
                clock_at_ref.append(cref)
                eref = int(self.eref) % (1<<16)
                encoder_at_ref.append(eref)
                self.eref += self.count_per_rev
            if c1 >= (1 << 32):
                c1 -= (1 << 32)
            if e1 >= (1 << 16):
                e1 -= (1 << 16)
                self.eref -= (1 << 16)
            self.t0 = t1
            self.c0 = c1
            self.e0 = e1
        frame['hwp_raw_clocks'] = clocks
        frame['hwp_raw_encoders'] = encoders
        frame['hwp_raw_clock_at_ref'] = clock_at_ref
        frame['hwp_raw_encoder_at_ref'] = encoder_at_ref
        frame['hwp_raw_time_at_1s'] = time_at_1s
        frame['hwp_raw_clock_at_1s'] = clock_at_1s
        return

def unwrap(x, nbit=32):
    ''' Unwrap overflow '''
    nrev = np.cumsum(np.append(
        0, np.diff(x.astype(np.int64)) < 0))
    return (nrev << nbit) + x

class Decode_HWP(object):
    ''' Calculate HWP angle from raw encoder data '''
    def __init__(self,
                 count_per_rev = 50000,
                 ref_offset = 0. * u.deg,
                 remove_raw = True
    ):
        self.count_per_rev = count_per_rev
        self.obs_start = None
        self.obs_end = None
        self.remove_raw = remove_raw

    def __call__(self, frame):
        if frame.type == core.G3FrameType.Observation:
            self.obs_start = int(frame['obs_start'])
            self.obs_end = int(frame['obs_end'])
        if frame.type != core.G3FrameType.Scan:
            return
        assert 'det_timestream' in frame

        det_n_sample = frame['det_timestream'].n_samples
        det_start = int(frame['det_timestream'].start)
        det_stop = int(frame['det_timestream'].stop)

        clks = np.array(frame['hwp_raw_clocks'], dtype=np.uint32)
        clks = unwrap(clks, 32)
        encs = np.array(frame['hwp_raw_encoders'], dtype=np.uint16)
        encs = unwrap(encs, 16)

        clk_ref = np.array(frame['hwp_raw_clock_at_ref'], dtype=np.uint32)
        clk_ref = unwrap(clk_ref, 32)
        enc_ref = np.array(frame['hwp_raw_encoder_at_ref'], dtype=np.uint16)
        enc_ref = unwrap(enc_ref, 16)

        time_1s = np.array(frame['hwp_raw_time_at_1s'], dtype=np.uint64)
        clk_1s = np.array(frame['hwp_raw_clock_at_1s'], dtype=np.uint32)
        clk_1s = unwrap(clk_1s, 32)

        if enc_ref.size >= 2:
            enc_ref_approx = np.interp(clk_ref, clks, encs).astype(int)
            if (enc_ref_approx - enc_ref > (1<<16) - 100).all():
                enc_ref += 1<<16
            enc_ref0 = np.mean(enc_ref % self.count_per_rev)
            self.enc_ref0 = frame['hwp_raw_encoder_at_ref'][-1] % self.count_per_rev
        else:
            enc_ref0 = self.enc_ref0

        if clk_1s.size >= 2:
            clock = (clk_1s[-1] - clk_1s[0]) / (time_1s[-1] - time_1s[0])
            enc_time = np.interp(
                clks - clk_1s[0],
                clk_1s - clk_1s[0],
                ((time_1s - time_1s[0]).astype(np.int64)
                 - (clk_1s - clk_1s[0]) / clock)).astype(np.int64)
            enc_time += ((clks - clk_1s[0]) / clock).astype(np.int64)
            enc_time = enc_time.astype(np.uint64) + time_1s[0]
            self.clock = clock
            self.clk0 = clk_1s[-1]
            self.t0 = time_1s[-1]
        else:
            clock = self.clock
            enc_time = ((clks - self.clk0) / clock).astype(np.int64)
            enc_time += self.t0
        enc_speed = (encs[-1] - encs[0]) / (clks[-1] - clks[0])
        enc_const = encs[0] + enc_speed * (clks - clks[0])
        encs_err = encs - enc_const

        det_time = np.linspace(det_start, det_stop, det_n_sample).astype(np.int64)
        enc_angle = np.interp(det_time, enc_time, encs_err)
        enc_angle += enc_speed * clock * (det_time - enc_time[0])
        enc_angle += encs[0] - enc_ref0
        enc_angle = (enc_angle % self.count_per_rev) / self.count_per_rev * 360 * u.deg
        enc_angle = core.G3Timestream(enc_angle)
        enc_angle.start = core.G3Time(det_start)
        enc_angle.stop = core.G3Time(det_stop)
        if self.remove_raw:
            for key in frame:
                if key.startswith('hwp_raw'):
                    del(frame[key])
        frame['hwp_angle'] = enc_angle
        return [frame]

''' Input values of HWP synchronous signals (HWPSS) for simulation '''
hwpss_frame = core.G3Frame(core.G3FrameType.Calibration)
hwpss_dict = core.G3MapDouble()
hwpss_dict['det1'] = 0.1 * u.K
hwpss_dict['det2'] = 0.1 * u.K
hwpss_frame['hwp_sim_hwpss'] = hwpss_dict

class Sim_HWPSS(object):
    ''' Inject HWPSS into detector timestreams '''
    def __init__(self, hwpss_frame=None):
        self.det_frame = None
        self.hwpss_frame = hwpss_frame

    def __call__(self, frame):
        if (frame.type == core.G3FrameType.Calibration and
            'det_list' in frame):
            self.det_frame = frame
            if self.hwpss_frame is not None:
                return [frame, self.hwpss_frame]
        if (frame.type == core.G3FrameType.Calibration and
            'hwp_sim_hwpss' in frame):
            self.hwpss_frame = frame
            return
        if frame.type != core.G3FrameType.Scan:
            return
        hwp_angle = np.asarray(frame['hwp_sim_true'])
        cos4f = np.cos(hwp_angle * 4)
        for det in self.det_frame['det_list']:
            hwpss = self.hwpss_frame['hwp_sim_hwpss'][det]
            hwpss_timestream = cos4f * hwpss
            frame['det_timestream'][det] += hwpss_timestream
        return

def G3pad(ts, pad, method='edge'):
    ''' Extend G3Timestreams '''
    start = int(ts.start)
    stop = int(ts.stop)
    n = ts.n_samples
    pad0, pad1 = pad
    start2 = start - int((stop - start) / (n-1) * pad0)
    stop2 = stop + int((stop - start) / (n-1) * pad1)
    x = np.asarray(ts)
    if method == 'hwp_encoder':
        slope = sum(np.diff(x)%(360. * u.deg)) / (stop - start)
        x2 = np.pad(x, pad, 'edge')
        x2[:pad0] = slope * np.linspace(start2 - start, 0, pad0 + 1)[:-1] + x[0]
        x2[pad0 + n:] = slope * np.linspace(0, stop2 - stop, pad1 + 1)[1:] + x[-1]
        x2 %= 360. * u.deg
    else:
        x2 = np.pad(x, pad, method)
    ts2 = core.G3Timestream(x2)
    ts2.start = core.G3Time(start2)
    ts2.stop = core.G3Time(stop2)
    return ts2

class Pad_Frame(object):
    ''' Extend Timestream in each frame '''
    def __init__(self,
                 target=['det_timestream', 'hwp_angle'],
                 pad = 4095, method='edge'):
        if isinstance(target, str):
            target = [target]
        self.target = target
        self.pad = pad
        if isinstance(method, str):
            method = [method] * len(target)
        self.method = method
        self.frame0 = None

    def __call__(self, frame):
        if frame.type != core.G3FrameType.Scan:
            if (self.frame0 is None or
                self.frame0.type != core.G3FrameType.Scan):
                self.frame0 = frame
                return
            for key, method in zip(self.target, self.method):
                if isinstance(self.frame0[key], core.G3Timestream):
                    ts2 = G3pad(self.frame0[key], (0, self.pad), method)
                elif isinstance(self.frame0[key], core.G3TimestreamMap):
                    ts2 = core.G3TimestreamMap()
                    for det in self.frame0[key].keys():
                        ts2[det] = G3pad(
                            self.frame0[key][det], (0, self.pad), method)
                del(self.frame0[key])
                self.frame0[key] = ts2
            frame0 = self.frame0
            self.frame0 = frame
            return [frame0, frame]
        if (self.frame0 is None or
            self.frame0.type != core.G3FrameType.Scan):
            for key, method in zip(self.target, self.method):
                if isinstance(frame[key], core.G3Timestream):
                    ts2 = G3pad(frame[key], (self.pad, 0), method)
                elif isinstance(frame[key], core.G3TimestreamMap):
                    ts2 = core.G3TimestreamMap()
                    for det in frame[key].keys():
                        ts2[det] = G3pad(
                            frame[key][det], (self.pad, 0), method)
                del(frame[key])
                frame[key] = ts2
            self.frame0 = frame
            return []
        for key, method in zip(self.target, self.method):
            if isinstance(frame[key], core.G3Timestream):
                if frame[key].n_samples >= self.pad:
                    ts0 = core.G3Timestream.concatenate([
                        self.frame0[key], frame[key][:self.pad]])
                else:
                    ts0 = G3pad(self.frame0[key],
                                (0, self.pad), method)
                if self.frame0[key].n_samples >= self.pad:
                    ts2 = core.G3Timestream.concatenate([
                        self.frame0[key][-self.pad:], frame[key]])
                else:
                    ts2 = G3pad(frame[key], (self.pad, 0), method)
            elif isinstance(frame[key], core.G3TimestreamMap):
                ts0 = core.G3TimestreamMap()
                ts2 = core.G3TimestreamMap()
                if frame[key].n_samples >= self.pad:
                    for det in frame[key].keys():
                        ts0[det] = core.G3Timestream.concatenate([
                            self.frame0[key][det],
                            frame[key][det][:self.pad]])
                else:
                    for det in frame[key].keys():
                        ts0[det] = G3pad(self.frame0[key][det],
                                         (0, self.pad), method)
                if self.frame0[key].n_samples >= self.pad:
                    for det in frame[key].keys():
                        ts2[det] = core.G3Timestream.concatenate([
                            self.frame0[key][det][-self.pad:],
                            frame[key][det]])
                else:
                    for det in frame[key].keys():
                        ts2[det] = G3pad(frame[key][det],
                                         (self.pad, 0), method)
            del(self.frame0[key])
            del(frame[key])
            self.frame0[key] = ts0
            frame[key] = ts2
        frame0 = self.frame0
        self.frame0 = frame
        return [frame0]

class Trim_Frame(object):
    ''' Trim Timestream in each frame '''
    def __init__(self,
                 target=['det_timestream', 'hwp_angle'],
                 trim = 4095):
        if isinstance(target, str):
            target = [target]
        self.target = target
        self.trim = trim

    def __call__(self, frame):
        if frame.type != core.G3FrameType.Scan:
            return
        for target in self.target:
            if isinstance(frame[target], core.G3Timestream):
                ts = frame[target][self.trim:-self.trim]
                del(frame[target])
                frame[target] = ts
            elif isinstance(frame[target], core.G3TimestreamMap):
                ts = core.G3TimestreamMap()
                for det in frame[target].keys():
                    ts[det] = frame[target][det][self.trim:-self.trim]
                del(frame[target])
                frame[target] = ts
        return [frame]

NFFTlist = np.array([2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,27,32,40,45,48,50,54,64,75,80,81,96,125,128,135,160,162,192,243,250,256,320,375,384,405,486,512,625,640,729,768,1024,1215,1250,1280,1458,1536,1875,2048,2187,2560,3072,3125,3645,4096,4374,5120,6144,6250,6561,8192,9375,10240,10935,12288,13122,15625,16384,19683,20480,24576,31250,32768,32805,39366,40960,46875,49152,59049,65536])

def getNFFT(n):
    ''' Find good length for FFT '''
    if n <= NFFTlist[-1]:
        return NFFTlist[np.searchsorted(NFFTlist,n)]
    else:
        return 1 << int(np.ceil(np.log2(n)))

def firwinc(numtaps, cutoff, width=None, window='hamming', pass_zero=True,
            scale=True, nyq=1.0):
    """
    Modification of scipy.signal.firwin to support complex window.
    """

    from scipy.signal import get_window, kaiser_atten, kaiser_beta

    cutoff = np.atleast_1d(cutoff) / float(nyq)

    # Check for invalid input.
    if cutoff.ndim > 1:
        raise ValueError("The cutoff argument must be at most "
                         "one-dimensional.")
    if cutoff.size == 0:
        raise ValueError("At least one cutoff frequency must be given.")
    if np.abs(cutoff).max() >= 1:
        raise ValueError("Invalid cutoff frequency: frequencies must be "
                         "greater than -nyq and less than nyq.")
    if np.any(np.diff(cutoff) <= 0):
        raise ValueError("Invalid cutoff frequencies: the frequencies "
                         "must be strictly increasing.")

    if width is not None:
        # A width was given.  Find the beta parameter of the Kaiser window
        # and set `window`.  This overrides the value of `window` passed in.
        atten = kaiser_atten(numtaps, float(width) / nyq)
        beta = kaiser_beta(atten)
        window = ('kaiser', beta)

    i0 = np.searchsorted(cutoff, 0., side='right')
    pass_bands = ((i0 + np.arange(cutoff.size + 1)) & 1) ^ pass_zero

    pass_nyquist = pass_bands[0] or pass_bands[-1]
    if pass_nyquist and numtaps % 2 == 0:
        raise ValueError("A filter with an even number of coefficients must "
                         "have zero response at the Nyquist rate.")

    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    cutoff = np.hstack(([-1.0] * pass_bands[0], cutoff, [1.0] * pass_bands[-1]))

    # `bands` is a 2D array; each row gives the left and right edges of
    # a passband.
    bands = cutoff.reshape(-1, 2)

    # Build up the coefficients.
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    f = fft.fftfreq(numtaps)
    h = np.zeros(numtaps, dtype=np.complex)
    for left, right in bands:
        h[(f * 2 >= left) * (f * 2 <= right)] = 1.
    h *= np.exp(- 2.j * np.pi * alpha * f)
    h = fft.ifft(h)

    # Get and apply the window function.
    win = get_window(window, numtaps, fftbins=False)
    h *= win

    # Now handle scaling if desired.
    if scale:
        # Get the first passband.
        left, right = bands[0]
        if left == 0:
            scale_frequency = 0.0
        elif right == 1:
            scale_frequency = 1.0
        else:
            scale_frequency = 0.5 * (left + right)
        c = np.cos(np.pi * m * scale_frequency)
        s = np.sum(h * c)
        h /= s

    return h

class Demodulate_HWP(object):
    ''' Demodulate detector timestreams '''
    def __init__(self,
                 target = 'det_timestream',
                 mode = [4],
                 bpf = [1.0 * u.Hz],
                 numtaps = 4095,
                 keep_drift = True
    ):
        self.target = target
        self.mode = mode
        self.bpf = bpf
        self.numtaps = numtaps
        self.keep_drift = keep_drift

    def __call__(self, frame):
        if frame.type != core.G3FrameType.Scan:
            return
        assert self.target in frame
        target = self.target
        assert 'hwp_angle' in frame
        hwp_ts = frame['hwp_angle']
        n_sample = hwp_ts.n_samples
        hwp_angle = np.asarray(hwp_ts)
        start = int(hwp_ts.start)
        stop = int(hwp_ts.stop)
        diff_angle = np.diff(hwp_angle) % (360. * u.deg)
        speed = (np.sum(diff_angle) / (stop - start)) / (360. * u.deg)
        nyq = hwp_ts.sample_rate / 2.
        n = self.numtaps
        fftsize = getNFFT(n_sample + n - 1)
        for mode, band in zip(self.mode, self.bpf):
            pass_zero = (speed * mode - band < 0.)
            bpf = firwinc(self.numtaps,
                          [-speed * mode - band,
                           -speed * mode + band],
                          nyq = nyq, pass_zero = pass_zero)
            fbpf = fft.fft(bpf, fftsize)
            exp_angle = np.exp(1.j * mode * hwp_angle)
            if mode == 0:
                out_ts = core.G3TimestreamMap()
            else:
                out_ts_r = core.G3TimestreamMap()
                out_ts_i = core.G3TimestreamMap()
            for det in frame[target].keys():
                x = np.asarray(frame[target][det])
                c = np.zeros(fftsize, np.complex)
                c[:n_sample] = x
                fft.fft(c, fftsize, overwrite_x=True)
                c *= fbpf
                fft.ifft(c, fftsize, overwrite_x=True)
                c = c[(n - 1) // 2 : (n - 1) // 2 + n_sample]
                if mode == 0:
                    c = c.real
                    sbtr_key = target + '_demod0_subtracted'
                    if self.keep_drift and sbtr_key in frame:
                        c += frame[sbtr_key][det]
                    out_ts[det] = core.G3Timestream(c)
                else:
                    c *= exp_angle * 2
                    cr = c.real
                    ci = c.imag
                    sbtr_key_r = target + '_demod%d_real_subtracted'%mode
                    sbtr_key_i = target + '_demod%d_imag_subtracted'%mode
                    if self.keep_drift and sbtr_key_r in frame:
                        cr += frame[sbtr_key_r][det]
                        ci += frame[sbtr_key_i][det]
                    out_ts_r[det] = core.G3Timestream(cr)
                    out_ts_i[det] = core.G3Timestream(ci)
            if mode == 0:
                out_ts.start = frame[target].start
                out_ts.stop = frame[target].stop
                outkey = target+'_demod0'
                if outkey in frame:
                    del(frame[outkey])
                frame[outkey] = out_ts
            else:
                outkey_r = target + '_demod%d_real' % mode
                outkey_i = target + '_demod%d_imag' % mode
                if outkey_r in frame:
                    del(frame[outkey_r], frame[outkey_i])
                out_ts_r.start = frame[target].start
                out_ts_r.stop = frame[target].stop
                frame[outkey_r] = out_ts_r
                out_ts_i.start = frame[target].start
                out_ts_i.stop = frame[target].stop
                frame[outkey_i] = out_ts_i
        return frame

def trend(x, mask=slice(None), method='linear'):
    ''' Return linear trend in the timestream '''
    polyorder = {'constant':0, 'linear':1}[method]
    t = np.linspace(-1, 1, len(x))
    p = np.polyfit(t[mask], x[mask], polyorder)
    return np.polyval(p, t)

class Subtract_HWPSS_Detrend(object):
    """
    Subtract HWPSS using the stable component of
    pre-demodulated timestreams
    """
    def __init__(self,
                 target = ['det_timestream'],
                 mode = [4],
                 method = 'linear',
                 mask = None):
        if isinstance(target, str):
            target = [target]
        self.target = target

        if isinstance(mode, int):
            mode = [mode]
        self.mode = mode

        if isinstance(method, str):
            method = [method] * len(mode)
        self.method = method
        if mask is None:
            self.mask = slice(None)
        else:
            self.mask = slice(mask, -mask)

    def __call__(self, frame):
        if frame.type != core.G3FrameType.Scan:
            return

        hwp_ts = frame['hwp_angle']
        hwp_angle = np.asarray(hwp_ts)
        t = np.array([x.time for x in hwp_ts.times()])
        cosmod = {}
        sinmod = {}
        for mode in self.mode:
            if mode == 0:
                cosmod[mode] = 2.
            else:
                cosmod[mode] = np.cos(hwp_angle * mode)
                sinmod[mode] = np.sin(hwp_angle * mode)

        for target in self.target:
            ts_subtract = {}
            for mode in self.mode:
                if mode == 0:
                    key = target + '_demod0_subtracted'
                    ts_subtract[key] = core.G3TimestreamMap()
                else:
                    keyreal = target + '_demod%d_real_subtracted'%mode
                    keyimag = target + '_demod%d_imag_subtracted'%mode
                    ts_subtract[keyreal] = core.G3TimestreamMap()
                    ts_subtract[keyimag] = core.G3TimestreamMap()

            for det in frame[target].keys():
                data = frame[target][det]
                for mode, method in zip(self.mode, self.method):
                    if mode == 0:
                        keys = [target + '_demod0']
                    else:
                        keys = [target + '_demod%d_real'%mode,
                                target + '_demod%d_imag'%mode]
                    for key in keys:
                        mod = sinmod if key.endswith('imag') else cosmod
                        demod = frame[key][det]
                        demod = trend(demod, self.mask, method)
                        data -= demod * mod[mode] / 2.
                        key += '_subtracted'
                        if key in frame:
                            demod += frame[key][det]
                        demod = core.G3Timestream(demod)
                        ts_subtract[key][det] = demod

            for key in ts_subtract.keys():
                ts_subtract[key].start = frame[target].start
                ts_subtract[key].stop = frame[target].stop
                if key in frame:
                    del(frame[key])
                frame[key] = ts_subtract[key]

if __name__ == '__main__':
    numtaps = 4095

    def add_slope(frame, start):
        if not 'det_timestream' in frame:
            return
        start = int(start)
        t = np.array([x.time for x in frame['det_timestream'].times()], dtype=np.int)
        for det in frame['det_timestream'].keys():
            ts2 = np.asarray(frame['det_timestream'][det])
            ts2 += (t - start) / u.s
            ts2 = core.G3Timestream(ts2)
            ts2.start = frame['det_timestream'][det].start
            ts2.stop = frame['det_timestream'][det].stop
            del(frame['det_timestream'][det])
            frame['det_timestream'][det] = ts2

    def print_frame(frame):
        print(frame)

    print('### Create simulation ###')
    start = core.G3Time('29-Mar-2019:01:12:06.345958000')
    end = start + int(179. * u.s)
    pipe = core.G3Pipeline()
    pipe.Add(Sim_Obs_Setup(obs_start = start, obs_end = end))
    pipe.Add(Sim_Detector_Setup())
    pipe.Add(Sim_Scan_Prepare(interval = 60.0 * u.s))
    pipe.Add(add_slope, start = start)
    pipe.Add(Sim_HWP_Raw(speed = 2.1 * u.Hz))
    pipe.Add(Sim_HWPSS(hwpss_frame))
    pipe.Add(print_frame)
    out = []
    pipe.Add(G3OutputFrameList(out))
    pipe.Run()
    print('### Done ###\n\n\n')

    print('### Analyze simulated data ###')
    pipe = core.G3Pipeline()
    pipe.Add(G3InputFrameList(out))
    pipe.Add(Decode_HWP())
    pipe.Add(Pad_Frame, target=['det_timestream', 'hwp_angle'],
             method=['edge', 'hwp_encoder'], pad=numtaps)
    pipe.Add(Demodulate_HWP, mode=[0, 4], bpf=[1.0 * u.Hz] * 2)
    pipe.Add(Subtract_HWPSS_Detrend, mode=[0, 4], mask=numtaps)
    pipe.Add(Demodulate_HWP, mode=[0, 4], bpf=[1.0 * u.Hz] * 2,
             keep_drift = False)
    pipe.Add(Trim_Frame,
             target=['det_timestream', 'hwp_angle',
                     'det_timestream_demod0',
                     'det_timestream_demod4_real',
                     'det_timestream_demod4_imag',
             ],
             trim=numtaps)
    pipe.Add(print_frame)
    pipe.Run()
