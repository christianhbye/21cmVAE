# Very Accurate Emulator

To install:
```
git clone https://github.com/christianhbye/21cmVAE.git
'''

```python
import 21cmVAE

my21cmVAE_class = VeryAccurateEmulator()  # the pretrained emulator

# make a list of parameters in the order ['fstar', 'Vc', 'fx', 'tau', 'alpha', 'nu_min', 'Rmfp']
# if you forget the order, do:
print(my21cmVAE_class.par_labels)

params = [0.0003, 4.2, 0, 0.055, 1.0, 0.1, 10]

global_signal = my21cmVAE_class.predict(params)

import matplotlib.pyplot as plt
redshifts = my21cmVAE_class.z_sampling
plt.figure()
plt.plot(redshifts, global_signal)
plt.title('Global signal vs redshift')
plt.xlabel('z')
plt.ylabel('T [mK]')
plt.show()

frequencies = my21cmVAE_class.nu_sampling
plt.figure()
plt.plot(frequencies, global_signal)
plt.title('Global signal vs frequency')
plt.xlabel(r'$\nu$ [MHz]')
plt.ylabel('T [mK]')
plt.show()
'''
