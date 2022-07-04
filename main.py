# by Tiszai Istvan, GPL-3.0 license
import os
import sys
import torch
from tkinter import *
from utils.device import select_device
from GUIs import mainGUI
sys.path.append(os.path.realpath('.'))

if __name__ == '__main__':
    dcount = torch.cuda.device_count()
    if (dcount > 0) :              
        select_device(device='0') 
        root = Tk()
        mainGUI(root)
        root.mainloop()