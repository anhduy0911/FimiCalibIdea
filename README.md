# Fimi Calibration - Single and Multidevice Calibration
## Dataset
```
Data/fimi_resample/envitus_fimi_overlapped.csv
```
## Main codebase

### Single Calib
You can take the components, as well as models from the base file of single model:
```
#run training model with default settings in forgan:
python forgan.py
```
Modules for the single calib tasks:
```
components.py
```

### Multidevice Calib
You can take the components, as well as models from the base file of multicalib model:
```
#run training model with default settings:
python main.py --usen 2

#training loop for multicalib model:
multicalib.py

#modules for the model:
models/modules.py
```
The same set for N model setting and 1 model setting
```
#for N models for N tasks
python main.py --usen 0 

#for 1 model for N tasks
python main.py --usen 1
```

## Environment
- Python
- Pytorch