import os
import numpy as np
import datetime as dt
from datetime import datetime
from Labber import ScriptTools
import time
import sys

def Wait(sec, remind_interval=1):
    num_10sec = sec // remind_interval
    rest_sec = sec % remind_interval
    for i in range(num_10sec):
        sys.stdout.write("\r")
        sys.stdout.write('Continue after ' + str(dt.timedelta(seconds=sec - remind_interval * i)))
        sys.stdout.flush()
        time.sleep(remind_interval)
    time.sleep(rest_sec)
    sys.stdout.write("\r")
    sys.stdout.write('Countdown ended\n')
    sys.stdout.flush()


def RunMeasurement(ConfigName, MeasLabel, ItemDict={}, ExePath='C:\Program Files\Labber\Program',
                   DataFolderName='Z:\Projects\Fluxonium\Data\\AugustusXVII', ConfigPath='', PrintOutFileName=False):
    # set path to executable
    ScriptTools.setExePath(ExePath)

    TimeNow = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    TimeStr = str(TimeNow)
    Year = TimeStr[:4]
    Month = TimeStr[5:7]
    Day = TimeStr[8:10]
    # define measurement objects
    # ConfigPath = 'C:\SC Lab\GitHubRepositories\measurement-with-labber\measurement setting/'
    DataPath = DataFolderName + '/' + Year + '/' + Month + '/' + 'Data_' + Month + Day + '/'

    if not os.path.exists(DataPath):
        os.makedirs(DataPath)

    ConfigFile = ConfigPath + ConfigName
    OutputFile = DataPath + MeasLabel + '_' + TimeStr
    if PrintOutFileName:
        print(MeasLabel + '_' + TimeStr)
    MeasObj = ScriptTools.MeasurementObject(ConfigFile, OutputFile)

    for item, value in ItemDict.items():
        if type(value) is list:
            for subitem in value:
                MeasObj.updateValue(item, subitem[0], itemType=subitem[1])
        else:
            MeasObj.updateValue(item, value)
    # if second_channel != '':
    #     MeasObj.setMasterChannel(second_channel)

    MeasObj.performMeasurement(return_data=False, use_scheduler=False)

    return [DataPath, MeasLabel + '_' + TimeStr + '.hdf5']
