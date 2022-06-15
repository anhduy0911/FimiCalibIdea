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
python main.py

#training loop for multicalib model:
multicalib.py

#modules for the model:
models/modules.py
```
## Environment
- Python
- Pytorch