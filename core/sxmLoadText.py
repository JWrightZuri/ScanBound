import os as os
# from core import nanonispy as nap
# import nanonis_load as sxm
import pySPM as spm
import matplotlib.pyplot as plt

dir = r'C:\Users\wrja\Desktop\STM_Data\Orito_Reaction\QPlus\20250803_PdGaA111_279Prod_2min_35C_98K'

for filename in os.listdir(dir):
    if filename.lower().endswith('.sxm'):
        filepath = os.path.join(dir, filename)
        imageData = spm.SXM(filepath)
        imageData.list_channels()
        fig, ax = plt.subplots(1,2,figsize=(14,7))
        imageData.get_channel('Z').show(ax=ax[0])
        p = imageData.get_channel('Current').show(ax=ax[1], cmap='viridis')
        # imageData.fft()
        plt.savefig(os.path.join(dir, filename.replace('.sxm', '.jpg')), bbox_inches='tight', pad_inches=0, dpi=500)