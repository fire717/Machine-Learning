
#y: one-hot numpy array
#e.g. [[1,0,0],[0,1,0]]
label_smoothing = 0.01
y = y * (1 - label_smoothing) + label_smoothing / num_classes
