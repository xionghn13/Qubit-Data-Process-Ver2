import numpy as np
import h5py
import Labber
import datetime


def readFSweepDat(file):
    Background = np.loadtxt(file, skiprows=1)
    BackFreq = Background[:, 0]
    BackPhase = np.unwrap(Background[:, 1])
    BackAmp = Background[:, 2]
    BackComplex = BackAmp * np.exp(1j * BackPhase)
    return [BackFreq, BackComplex]


def readFSweepLabber(file):
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    BackFreq_var = 'Qubit - Frequency'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if BackFreq_var not in Entries:
        BackFreq_var = 'Readout - Frequency'
    BackComplex = np.conj(LogData.getData(ATS_var)[0])
    BackFreq = LogData.getData(BackFreq_var)[0] * 1e-9
    return [BackFreq, BackComplex]


def readFISweepDat(file):
    # for HQC data
    OneTone = np.loadtxt(file, skiprows=1)
    OneCurrent = OneTone[:, 0]
    OneCurrentUniq = np.unique(OneCurrent)
    NumCurrent = np.size(OneCurrentUniq)
    OneFreq = OneTone[:, 1]
    OneFreqUniq = np.unique(OneFreq)
    NumFreq = np.size(OneFreqUniq)
    if NumCurrent * NumFreq > np.size(OneCurrent):
        NumCurrent = NumCurrent - 1
        OneCurrent = OneCurrent[0:NumCurrent * NumFreq]
        OneFreq = OneFreq[0:NumCurrent * NumFreq]
        OneTone = OneTone[0:NumCurrent * NumFreq, :]

    OneCurrent = np.transpose(np.reshape(OneCurrent, (NumCurrent, NumFreq)))
    OneFreq = np.transpose(np.reshape(OneFreq, (NumCurrent, NumFreq)))
    OnePhase = np.unwrap(np.transpose(np.reshape(OneTone[:, 2], (NumCurrent, NumFreq))), axis=0)
    OneAmp = np.transpose(np.reshape(OneTone[:, 3], (NumCurrent, NumFreq)))
    OneComplex = OneAmp * np.exp(1j * OnePhase)
    return [OneFreq, OneCurrent, OneComplex]


def readFISweepLabber(file):
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    OneFreq_var = 'Qubit - Frequency'
    Current_var = 'Yoko - Current'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if OneFreq_var not in Entries:
        OneFreq_var = 'Readout - Frequency'
    OneComplex = np.conj(np.transpose(LogData.getData(ATS_var)))
    OneFreq = np.transpose(LogData.getData(OneFreq_var)) * 1e-9
    OneCurrent = np.transpose(LogData.getData(Current_var)) * 1000
    return [OneFreq, OneCurrent, OneComplex]


def readFISweepTwoToneLabber(file):
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    OneFreq_var = 'Pump - Frequency'
    Current_var = 'Yoko - Current'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if OneFreq_var not in Entries:
        OneFreq_var = 'Drive2 - Frequency'
    OneComplex = np.conj(np.transpose(LogData.getData(ATS_var)))
    OneFreq = np.transpose(LogData.getData(OneFreq_var)) * 1e-9
    OneCurrent = np.transpose(LogData.getData(Current_var)) * 1000
    return [OneFreq, OneCurrent, OneComplex]


def readFSweepTwoToneLabber(file):
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    OneFreq_var = 'Pump - Frequency'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if OneFreq_var not in Entries:
        OneFreq_var = 'Drive2 - Frequency'
    OneComplex = np.conj((LogData.getData(ATS_var)))[0]
    OneFreq = (LogData.getData(OneFreq_var))[0] * 1e-9
    return [OneFreq, OneComplex]


def readFPSweepDat(file):
    # for HQC data
    OneTone = np.loadtxt(file, skiprows=1)
    OnePower = OneTone[:, 0]
    OnePowerUniq = np.unique(OnePower)
    NumPower = np.size(OnePowerUniq)
    OneFreq = OneTone[:, 1]
    OneFreqUniq = np.unique(OneFreq)
    NumFreq = np.size(OneFreqUniq)
    if NumPower * NumFreq > np.size(OnePower):
        NumPower = NumPower - 1
        OneTone = OneTone[0:NumPower * NumFreq, :]
        OnePower = OneTone[:, 0]
        OneFreq = OneTone[:, 1]
    OnePower = np.transpose(np.reshape(OnePower, (NumPower, NumFreq)))
    OneFreq = np.transpose(np.reshape(OneFreq, (NumPower, NumFreq)))
    OnePhase = np.unwrap(np.transpose(np.reshape(OneTone[:, 2], (NumPower, NumFreq))), axis=0)
    OneAmp = np.transpose(np.reshape(OneTone[:, 3], (NumPower, NumFreq)))
    OneComplex = OneAmp * np.exp(1j * OnePhase)
    return [OneFreq, OnePower, OneComplex]


def readFPSweepLabber(file):
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    OneFreq_var = 'Qubit - Frequency'
    Power_var = 'Qubit - Power'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if Power_var not in Entries:
        Power_var = 'Readout - Power'
        OneFreq_var = 'Readout - Frequency'
    OneComplex = np.conj(np.transpose(LogData.getData(ATS_var)))
    OneFreq = np.transpose(LogData.getData(OneFreq_var)) * 1e-9
    OnePower = np.transpose(LogData.getData(Power_var))
    return [OneFreq, OnePower, OneComplex]


def readReadoutPowerLabber(file):
    Power_var = 'Qubit - Power'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if Power_var not in Entries:
        Power_var = 'Readout - Power'
    Power = np.unique(np.transpose(LogData.getData(Power_var)))
    if len(Power) == 1:
        Power = Power[0]
    return Power


def readSetup2ReadoutPowerLabber(file):
    Power_var = 'Cavity RF - Power'
    LogData = Labber.LogFile(file)
    Power = np.unique(np.transpose(LogData.getData(Power_var)))
    if len(Power) == 1:
        Power = Power[0]
    return Power


def readReadoutFreqLabber(file):
    Freq_var = 'Qubit - Frequency'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if Freq_var not in Entries:
        Freq_var = 'Readout - Frequency'
    Freq = np.unique(np.transpose(LogData.getData(Freq_var))) * 1e-9
    if len(Freq) == 1:
        Freq = Freq[0]
    return Freq


def readPumpPowerLabber(file):
    Power_var = 'Pump - Power'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if Power_var not in Entries:
        Power_var = 'Drive2 - Power'
    Power = np.unique(np.transpose(LogData.getData(Power_var)))
    if len(Power) == 1:
        Power = Power[0]
    return Power


def readDrive1PowerLabber(file):
    Power_var = 'Drive1 - Power'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    Power = np.unique(np.transpose(LogData.getData(Power_var)))
    if len(Power) == 1:
        Power = Power[0]
    return Power


def readPumpFreqLabber(file):
    Freq_var = 'Pump - Frequency'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if Freq_var not in Entries:
        Freq_var = 'Drive2 - Frequency'
    Freq = np.unique(np.transpose(LogData.getData(Freq_var))) * 1e-9
    if len(Freq) == 1:
        Freq = Freq[0]
    return Freq


def readCurrentLabber(file):
    Cur_var = 'Yoko - Current'
    LogData = Labber.LogFile(file)
    Cur = np.unique(np.transpose(LogData.getData(Cur_var))) * 1e3
    if len(Cur) == 1:
        Cur = Cur[0]
    return Cur


def readFirstPulseDelayLabber(file):
    Delay_var = 'Pulse Generator - First pulse delay'
    LogData = Labber.LogFile(file)
    Delay = np.unique(np.transpose(LogData.getData(Delay_var)))
    if len(Delay) == 1:
        Delay = Delay[0]
    return Delay


def getFolder(file, LabberFolder='C:\\Users/admin\Labber\Data/'):
    name_str_list = file.split('_')
    type_str = name_str_list[0]
    date_str_lsit = name_str_list[-1].split('-')
    year = date_str_lsit[0]
    month = date_str_lsit[1]
    day = date_str_lsit[2]
    file_folder = LabberFolder + year + '\\' + month + '\\' + 'Data_' + month + day + '\\'
    return file_folder


def readDrivingPulseLenLabber(file):
    PulseLen_var = 'Pulse Generator - Width #1'
    LogData = Labber.LogFile(file)
    PulseLen = np.unique(np.transpose(LogData.getData(PulseLen_var)))
    if len(PulseLen) == 1:
        PulseLen = PulseLen[0]
    return PulseLen


def readRabiH5(file):
    # for HQC data
    f = h5py.File(file, 'r')
    key_list = list(f.keys())

    # Get the data
    rabi_time = list(f[key_list[2]])[0]
    rabi_amp = list(f[key_list[0]])[0][0]
    rabi_phase = list(f[key_list[1]])[0][0]
    rabi_complex = rabi_amp * np.exp(1j * rabi_phase)
    rabi_complex1 = rabi_complex[0::2]
    rabi_complex2 = rabi_complex[1::2]
    return [rabi_time, rabi_complex1, rabi_complex2]


def readRabiLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Width #1'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    if Time_var not in Entries:
        Time_var = 'Pulse Generator - Width #2'
    rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, 0]
    rabi_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    if np.unique(rabi_time).__len__() == 1:
        Time_var = 'Pulse Generator - Plateau #1'
        rabi_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9

    return [rabi_time, rabi_complex]


def readMultiRabiLabber(file):
    # for Labber data
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Width #1'
    if Time_var not in Entries:
        Time_var = 'Pulse Generator - Plateau #1'
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, :]
    rabi_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    if np.unique(rabi_time).__len__() == 1:
        Time_var = 'Pulse Generator - Plateau #1'
        rabi_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9

    return [rabi_time, rabi_complex]

def readMultiRabiInterleavedLabber(file):
    # for Labber data
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Width #1'
    if Time_var not in Entries:
        Time_var = 'Pulse Generator - Plateau #1'
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    rabi_complex = np.conj((LogData.getData(ATS_var)))
    rabi_time = np.unique(LogData.getData(Time_var)) * 1e9
    if np.unique(rabi_time).__len__() == 1:
        Time_var = 'Pulse Generator - Plateau #1'
        rabi_time = np.unique(LogData.getData(Time_var)) * 1e9

    return [rabi_time, rabi_complex]



def readRepeatedRabiSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Width #1'
    Counter_var = 'Counter - Number of points'
    LogData = Labber.LogFile(file)
    t1_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
        t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    else:
        t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, ::len(t1_time)]
    t1_counter = np.unique(np.transpose(LogData.getData(Counter_var)))

    return [t1_time, t1_counter, t1_complex]


def readRabiCH1DriveLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Plateau #1'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, 0]
    rabi_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9

    return [rabi_time, rabi_complex]


def readRabiCH1PumpedLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Width #2'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, 0]
    rabi_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9

    return [rabi_time, rabi_complex]


def readRabiPowerSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Width #1'
    Power_var = 'Pump - Power'
    LogData = Labber.LogFile(file)
    rabi_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
        Power_var = 'Drive2 - Power'
        rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    else:
        rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, ::len(rabi_time)]

    rabi_power = LogData.getData(Power_var)[:, 0]
    return [rabi_time, rabi_power, rabi_complex]


def readRabiFreqSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Width #1'
    Freq_var = 'Qubit - Frequency'
    LogData = Labber.LogFile(file)
    rabi_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
        Freq_var = 'Readout - Frequency'
        rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    else:
        rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, ::len(rabi_time)]
    rabi_freq = np.unique(np.transpose(LogData.getData(Freq_var))) * 1e-9

    return [rabi_time, rabi_freq, rabi_complex]


def readT1H5(file):
    # for HQC data
    f = h5py.File(file, 'r')
    key_list = list(f.keys())

    # Get the data
    t1_time = list(f[key_list[0]])[0]
    t1_amp = list(f[key_list[1]])[0][0]
    t1_phase = list(f[key_list[2]])[0][0]
    t1_complex = t1_amp * np.exp(1j * t1_phase)
    t1_complex1 = t1_complex[0::2]
    t1_complex2 = t1_complex[1::2]
    return [t1_time, t1_complex1, t1_complex2]


def readT1Labber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Readout delay'
    # Time_var = 'Pulse Generator - Demodulation - Length'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, 0]
    t1_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9

    return [t1_time, t1_complex]


def readMultiT1Labber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Readout delay'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, :]
    t1_time = np.unique(LogData.getData(Time_var)) * 1e9

    return [t1_time, t1_complex]

def readMultiT1InterleavedLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Readout delay'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    t1_complex = np.conj((LogData.getData(ATS_var)))
    t1_time = np.unique(LogData.getData(Time_var)) * 1e9

    return [t1_time, t1_complex]

def readIntegratedT1Labber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    Time_var = 'Alazar - Number of samples'
    LogData = Labber.LogFile(file)
    t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, 0]
    t1_time = np.transpose(LogData.getData(Time_var))[:, 0]

    return [t1_time, t1_complex]


def readT1PowerSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Readout delay'
    Power_var = 'Pump - Power'
    LogData = Labber.LogFile(file)
    t1_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
        Power_var = 'Drive2 - Power'
        t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    else:
        t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, ::len(t1_time)]
    t1_power = np.unique(np.transpose(LogData.getData(Power_var)))
    return [t1_time, t1_power, t1_complex]


def readT1ReadoutPowerSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Readout delay'
    Power_var = 'Qubit - Power'
    LogData = Labber.LogFile(file)
    t1_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
        Power_var = 'Readout - Power'
        t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    else:
        t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, ::len(t1_time)]
    t1_power = np.unique(np.transpose(LogData.getData(Power_var)))
    return [t1_time, t1_power, t1_complex]


def readRepeatedT1SweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Readout delay'
    Counter_var = 'Counter - Number of points'
    LogData = Labber.LogFile(file)
    t1_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
        t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    else:
        t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, ::len(t1_time)]
    t1_counter = np.unique(np.transpose(LogData.getData(Counter_var)))

    return [t1_time, t1_counter, t1_complex]


def readSetup2RepeatedT1SweepLabber(file):
    # for Labber data
    ATS_var = 'AlazarTech Signal Demodulator - Channel A - Average demodulated value'
    Time_var = 'Multi-Qubit Pulse Generator - Sequence duration'
    Counter_var = 'DummyVariable - Number of points'
    LogData = Labber.LogFile(file)
    t1_counter = np.unique(np.transpose(LogData.getData(Counter_var)))
    t1_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    return [t1_time, t1_counter, t1_complex]


def readRepeatedIntegratedT1SweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    Time_var = 'Alazar - Number of samples'
    Counter_var = 'Counter - Number of points'
    LogData = Labber.LogFile(file)
    t1_counter = np.unique(np.transpose(LogData.getData(Counter_var)))
    t1_time = np.transpose(LogData.getData(Time_var))[:, 0]
    t1_complex = np.conj(np.transpose(LogData.getData(ATS_var)))

    return [t1_time, t1_counter, t1_complex]


def readRepeatedFSweepTwoToneLabber(file, ATS_var='Alazar - Channel A - Average demodulated value',
                                    Freq_var='Pump - Frequency', Counter_var='Counter - Number of points'):
    # for Labber data
    LogData = Labber.LogFile(file)
    complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    freq = np.transpose(LogData.getData(Freq_var)) * 1e-9
    counter = np.transpose(LogData.getData(Counter_var))

    return [freq, counter, complex]


def readT2H5(file):
    # for HQC data
    f = h5py.File(file, 'r')
    key_list = list(f.keys())

    # Get the data
    t1_time = list(f[key_list[2]])[0]
    t1_amp = list(f[key_list[0]])[0][0]
    t1_phase = list(f[key_list[1]])[0][0]
    t1_complex = t1_amp * np.exp(1j * t1_phase)
    return [t1_time, t1_complex]


def readT2Labber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Sequence duration'
    LogData = Labber.LogFile(file)
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
    if Time_var not in Entries:
        Time_var = 'Pulse Generator - Demodulation - Length'
    t2_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, 0]
    t2_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9

    return [t2_time, t2_complex]


def readRepeatedT2SweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Sequence duration'
    Counter_var = 'Counter - Number of points'
    LogData = Labber.LogFile(file)
    t2_time = np.transpose(LogData.getData(Time_var))[:, 0] * 1e9
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
        t2_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    else:
        t2_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, ::len(t2_time)]
    t2_counter = np.unique(np.transpose(LogData.getData(Counter_var)))
    return [t2_time, t2_counter, t2_complex]


def readRepeatedT1T2InterleavedSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average buffer demodulated values'
    Time_var = 'Pulse Generator - Sequence duration'
    Counter_var = 'Counter - Number of points'
    LogData = Labber.LogFile(file)
    counter = np.unique(np.transpose(LogData.getData(Counter_var)))
    num_time = int(len(np.transpose(LogData.getData(Time_var))[0, :]) / len(counter))
    time = np.transpose(LogData.getData(Time_var))[0, :num_time] * 1e9
    Entries = LogData.getEntry()
    if ATS_var not in Entries:
        ATS_var = 'Alazar - Channel A - Average demodulated value'
        t1_t2_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
        t1_complex = t1_t2_complex[0, :].reshape((len(counter), num_time)).transpose()
        t2_complex = t1_t2_complex[1, :].reshape((len(counter), num_time)).transpose()
    else:
        t1_t2_complex = np.conj(np.transpose(LogData.getData(ATS_var)))[:, ::len(time) * 2]
        t1_complex = t1_t2_complex[0::2]
        t2_complex = t1_t2_complex[1::2]
    return [time, counter, t1_complex, t2_complex]


def readReferencedRepeatedT1T2InterleavedSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    Time_var = 'Pulse Generator - Sequence duration'
    Counter_var = 'Counter - Number of points'
    Freq_var = 'Readout - Frequency'
    Demod_Freq_var = 'Alazar - Demodulation frequency'
    # Power_var = 'Readout - Power'
    LogData = Labber.LogFile(file)
    # power = np.unique(np.transpose(LogData.getData(Power_var)))[0]
    freq = np.unique(np.transpose(LogData.getData(Freq_var)))[0] * 1e-9
    demod_freq = LogData.getChannelValue(Demod_Freq_var) * 1e-9
    freq = np.array([freq - demod_freq, freq + demod_freq])
    counter = np.unique(np.transpose(LogData.getData(Counter_var)))
    time = np.unique(LogData.getData(Time_var)) * 1e9
    num_time = len(time)
    # print(len(np.unique(LogData.getData(Time_var))))
    # time = np.transpose(LogData.getData(Time_var))[0, :num_time] * 1e9
    t1_t2_complex = np.conj(LogData.getData(ATS_var))
    print(t1_t2_complex.shape)
    ref_complex = t1_t2_complex[:, 1].reshape((len(counter), num_time, 2))
    t1_t2_complex = t1_t2_complex[:, 0].reshape((len(counter), num_time, 2))
    print(t1_t2_complex.shape)
    t1_complex = np.transpose(np.concatenate((t1_t2_complex[:, :, [0]], ref_complex[:, :, [0]]), axis=2), axes=[1, 0, 2])
    t2_complex = np.transpose(np.concatenate((t1_t2_complex[:, :, [1]], ref_complex[:, :, [1]]), axis=2), axes=[1, 0, 2])
    # print(t1_complex[0, 0, 0])
    return [time, counter, freq, t1_complex, t2_complex]


def readReferencedRepeatedT1RamseyEchoInterleavedSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    Time_var = 'Pulse Generator - Sequence duration'
    Counter_var = 'Counter - Number of points'
    Freq_var = 'Readout - Frequency'
    Demod_Freq_var = 'Alazar - Demodulation frequency'
    # Power_var = 'Readout - Power'
    LogData = Labber.LogFile(file)
    # power = np.unique(np.transpose(LogData.getData(Power_var)))[0]
    freq = np.unique(np.transpose(LogData.getData(Freq_var)))[0] * 1e-9
    demod_freq = LogData.getChannelValue(Demod_Freq_var) * 1e-9
    freq = np.array([freq - demod_freq, freq + demod_freq])
    counter = np.unique(np.transpose(LogData.getData(Counter_var)))
    time = np.unique(LogData.getData(Time_var)) * 1e9
    num_time = len(time)
    # print(len(np.unique(LogData.getData(Time_var))))
    # time = np.transpose(LogData.getData(Time_var))[0, :num_time] * 1e9
    t1_t2_complex = np.conj(LogData.getData(ATS_var))
    # print(t1_t2_complex.shape)
    ref_complex = t1_t2_complex[:, 1].reshape((len(counter), num_time, 3))
    t1_t2_complex = t1_t2_complex[:, 0].reshape((len(counter), num_time, 3))
    # print(t1_t2_complex.shape)
    t1_complex = np.transpose(np.concatenate((t1_t2_complex[:, :, [0]], ref_complex[:, :, [0]]), axis=2), axes=[1, 0, 2])
    ramsey_complex = np.transpose(np.concatenate((t1_t2_complex[:, :, [1]], ref_complex[:, :, [1]]), axis=2), axes=[1, 0, 2])
    echo_complex = np.transpose(np.concatenate((t1_t2_complex[:, :, [2]], ref_complex[:, :, [1]]), axis=2), axes=[1, 0, 2])
    # print(t1_complex[0, 0, :])
    return [time, counter, freq, t1_complex, ramsey_complex, echo_complex]


def readSetup2RepeatedT1T2InterleavedSweepLabber(file):
    # for Labber data
    ATS_var = 'Signal Demodulation - Value'
    Time_var = 'Multi-Qubit Pulse Generator - Sequence duration'
    Counter_var = 'Single-Qubit Pulse Generator - Number of points'
    LogData = Labber.LogFile(file)
    counter = np.unique(np.transpose(LogData.getData(Counter_var)))
    num_time = int(len(np.transpose(LogData.getData(Time_var))[0, :]) / len(counter))
    time = np.transpose(LogData.getData(Time_var))[0, :num_time] * 1e9
    t1_t2_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    t1_complex = t1_t2_complex[0, :].reshape((len(counter), len(time))).transpose()
    t2_complex = t1_t2_complex[1, :].reshape((len(counter), len(time))).transpose()
    return [time, counter, t1_complex, t2_complex]


def readRepeatedT1RamseyEchoInterleavedSweepLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    Time_var = 'Pulse Generator - Sequence duration'
    Counter_var = 'Counter - Number of points'
    LogData = Labber.LogFile(file)
    counter = np.unique(np.transpose(LogData.getData(Counter_var)))
    num_time = int(len(np.transpose(LogData.getData(Time_var))[0, :]) / len(counter))
    time = np.transpose(LogData.getData(Time_var))[0, :num_time] * 1e9
    t1_t2_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    t1_complex = t1_t2_complex[0, :].reshape((len(counter), num_time)).transpose()
    ramsey_complex = t1_t2_complex[1, :].reshape((len(counter), num_time)).transpose()
    echo_complex = t1_t2_complex[2, :].reshape((len(counter), num_time)).transpose()
    return [time, counter, t1_complex, ramsey_complex, echo_complex]


def readRefRabiCalLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    Freq_var = 'Readout - Frequency'
    Demod_Freq_var = 'Alazar - Demodulation frequency'
    Power_var = 'Readout - Power'
    LogData = Labber.LogFile(file)
    power = np.unique(np.transpose(LogData.getData(Power_var)))[0]
    freq = np.unique(np.transpose(LogData.getData(Freq_var)))[0] * 1e-9
    demod_freq = LogData.getChannelValue(Demod_Freq_var) * 1e-9
    freq = np.array([freq - demod_freq, freq + demod_freq])
    rabi_complex = np.mean(np.conj(np.transpose(LogData.getData(ATS_var))), axis=1)
    return [freq, power, rabi_complex]


def readRefRabiLabber(file):
    # for Labber data
    ATS_var = 'Alazar - Channel A - Average demodulated value'
    Freq_var = 'Readout - Frequency'
    Demod_Freq_var = 'Alazar - Demodulation frequency'
    Time_var = 'Pulse Generator - Width #1'
    LogData = Labber.LogFile(file)
    freq = np.unique(np.transpose(LogData.getData(Freq_var)))[0] * 1e-9
    demod_freq = LogData.getChannelValue(Demod_Freq_var) * 1e-9
    freq_low = freq - demod_freq
    freq_high = freq + demod_freq
    time = np.unique(np.transpose(LogData.getData(Time_var))) * 1e9
    rabi_complex = np.conj(np.transpose(LogData.getData(ATS_var)))
    rabi_lowf = rabi_complex[0, :]
    rabi_highf = rabi_complex[1, :]
    return [time, freq_low, freq_high, rabi_lowf, rabi_highf]


# read VNA data
def readVNA4portS11(file):
    # for Labber data
    S11_var = 'VNA-4port - S11'
    start_freq_var = 'VNA-4port - Start frequency'
    stop_freq_var = 'VNA-4port - Stop frequency'
    num_of_points_var = 'VNA-4port - # of points'
    LogData = Labber.LogFile(file)
    S11_complex = (np.transpose(LogData.getData(S11_var)))[:, 0]
    Start_freq = np.transpose(LogData.getData(start_freq_var))[:, 0] * 1e-9
    Stop_freq = np.transpose(LogData.getData(stop_freq_var))[:, 0] * 1e-9
    Num_of_points = len(S11_complex)
    freq = np.linspace(Start_freq, Stop_freq, Num_of_points)

    return [freq, S11_complex]


def readVNAS11(file):
    # for Labber data
    S11_var = 'VNA - S11'
    start_freq_var = 'VNA - Start frequency'
    stop_freq_var = 'VNA - Stop frequency'
    num_of_points_var = 'VNA - # of points'
    LogData = Labber.LogFile(file)
    S11_complex = (np.transpose(LogData.getData(S11_var)))[:, 0]
    Start_freq = np.transpose(LogData.getData(start_freq_var))[:, 0] * 1e-9
    Stop_freq = np.transpose(LogData.getData(stop_freq_var))[:, 0] * 1e-9
    Num_of_points = len(S11_complex)
    freq = np.linspace(Start_freq, Stop_freq, Num_of_points)

    return [freq, S11_complex]


def readVNA4portS21(file):
    # for Labber data
    S11_var = 'VNA-4port - S21'
    start_freq_var = 'VNA-4port - Start frequency'
    stop_freq_var = 'VNA-4port - Stop frequency'
    num_of_points_var = 'VNA-4port - # of points'
    LogData = Labber.LogFile(file)
    S11_complex = (np.transpose(LogData.getData(S11_var)))[:, 0]
    Start_freq = np.transpose(LogData.getData(start_freq_var))[:, 0] * 1e-9
    Stop_freq = np.transpose(LogData.getData(stop_freq_var))[:, 0] * 1e-9
    Num_of_points = len(S11_complex)
    freq = np.reshape(np.linspace(Start_freq, Stop_freq, Num_of_points), (Num_of_points,))

    return [freq, S11_complex]


def readVNAS21(file):
    # for Labber data
    S11_var = 'VNA - S21'
    start_freq_var = 'VNA - Start frequency'
    stop_freq_var = 'VNA - Stop frequency'
    num_of_points_var = 'VNA - # of points'
    LogData = Labber.LogFile(file)
    S11_complex = (np.transpose(LogData.getData(S11_var)))[:, 0]
    Start_freq = np.transpose(LogData.getData(start_freq_var))[:, 0] * 1e-9
    Stop_freq = np.transpose(LogData.getData(stop_freq_var))[:, 0] * 1e-9
    Num_of_points = len(S11_complex)
    S11_complex = np.reshape(S11_complex, (Num_of_points,))
    freq = np.reshape(np.linspace(Start_freq, Stop_freq, Num_of_points), (Num_of_points,))

    return [freq, S11_complex]


# read time
def readStartStopTime(file):
    LogData = Labber.LogFile(file)
    return [LogData.getEntry(0)['timestamp'], LogData.getEntry(-1)['timestamp']]


def getTimeStampArray(file, count):
    [start, stop] = readStartStopTime(file)
    return np.linspace(start, stop, count)


def SecArrayToDateTimeList(arr):
    l = []
    for sec in arr:
        l += [datetime.datetime.fromtimestamp(sec)]
    return l
