import Script_TwoTone
import Script_TimeDomain
from QubitDataProcessPackages import *

# CurrentList = np.array([2.545e-3])
CurrentList = np.concatenate((np.linspace(2.492e-3, 2.5e-3, 5), np.linspace(2.502e-3, 2.542e-3, 11)))
# CurrentList = np.concatenate((np.linspace(2.466e-3, 2.542e-3, 20), np.linspace(2.544e-3, 2.62e-3, 39)))

# CurrentList = np.linspace(2.59e-3, 2.62e-3, 16)
#
# CurrentList = np.insert(CurrentList, 0, 6.18e-3)
print(CurrentList)
Anchor1 = [2.488e-3, 548.5e6]
Anchor2 = [2.49e-3, 546.9e6]
AnchorReadout1 = [2.54e-3, 7.685e9]
AnchorReadout2 = [2.55e-3, 7.685e9]
SaveFig = True
Span = 20e6
DrivingPower = -15
PiPulseLength = 25e-9
ReadoutFreq = 7.962e9
ReadoutPower = 4
T1MaxDelay = 60e-6
PulseType = 0  # 0 Gaussian, 1 Square
Avg = 50e3
T2RamseyDetuning = 0.4e6
CyclePoints = 400e3
TwoTonePower = -25
TwoToneAvg = 50e3
TwoToneSeqLen = 10e3
# PiPulseLength = 500e-6
# MeasTypeList = ['rabi']
# MeasTypeList = ['t2_ramsey', 't1', 't2_echo']
# MeasTypeList = ['t1']
# MeasTypeList = ['t2_ramsey', 't2_echo']
# MeasTypeList = ['rabi', 't1', 't2_ramsey', 't2_echo']
# MeasTypeList = ['rabi', 't2_ramsey', 't2_echo']
# MeasTypeList = ['rabi', 't1', 't2_echo']
MeasTypeList = ['rabi', 't1']
# MeasTypeList = ['rabi', 't1_t2_interleaved']
# MeasTypeList = ['rabi', 't2_ramsey', 't1_t2_interleaved']
for i, cur in enumerate(CurrentList):
    [fc, width] = 1e9 * Script_TwoTone.FindReadoutFreqOneTone(Current, AnchorReadout1, AnchorReadout2,
                                                              ReadoutPower=ReadoutPower, SaveFig=SaveFig)
    AnchorReadout2 = AnchorReadout1
    AnchorReadout1 = [cur, fc]
    f0 = 1e9 * Script_TwoTone.FindQubitFreqTwoTone(cur, Anchor1, Anchor2, Power=TwoTonePower, ReadoutFreq=ReadoutFreq,
                                                   ReadoutPower=ReadoutPower, Avg=TwoToneAvg, Span=Span,
                                                   SeqLen=TwoToneSeqLen, SaveFig=SaveFig)
    Anchor2 = Anchor1
    Anchor1 = [cur, f0]
    print('Current = %.4GmA\nQubitFreq = %.4GGHz' % (cur * 1e3, f0 / 1e9))
    for MeasType in MeasTypeList:
        FitDict = Script_TimeDomain.timeDomainMeasurement(cur, ReadoutFreq, f0, DrivingPower, PiPulseLength,
                                                          T1MaxDelay=T1MaxDelay, Avg=Avg, Detuning=T2RamseyDetuning,
                                                          DutyCyclePoints=CyclePoints, ReadoutPower=ReadoutPower,
                                                          PulseType=PulseType, MeasurementType=MeasType,
                                                          FitAndPlot=True, ShowFig=False)
        if MeasType == 'rabi':
            PiPulseLength = round(FitDict['opt'][3]) * 1e-9
            print('PiPulseLength now is %.3Gs' % PiPulseLength)
            # elif MeasType == 't1':
            #     T1MaxDelay = round(FitDict['opt'][1] * 5 / 1e3) * 1e-6
            #     print('T1MaxDelay now is %.3Gs' % T1MaxDelay)
