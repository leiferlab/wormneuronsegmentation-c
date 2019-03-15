import numpy as np
import matplotlib.pyplot as plt
import time

folder = "/tigress/LEIFER/francesco/pumpprobe/mec-4/20190219/pumpprobe_20190219_090046/"

f = open(folder+'sCMOS_Frames_U16_1024x512.dat','br')
t0 = time.time()
A = np.copy(np.fromfile(f, dtype=np.uint16, count=512*1024).reshape((512,1024)).T)
print((time.time()-t0))
f.close()

plt.imshow(A)
plt.show()
