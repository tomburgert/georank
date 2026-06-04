import os
import numpy as np
import lmdb
import pandas as pd
import netCDF4
import xarray as xr

import sys
sys.path.append('/workspace/modules/mlc_noise/data')
from s2_interface import S4A_Patch


CROP_ENCODING = {
    'Cereals': 100,
    'Wheat': 110,
    'Maize': 120,
    'Rice': 130,
    'Sorghum': 140,
    'Barley': 150,
    'Rye': 160,
    'Oats': 170,
    'Millets': 180,
    'Other cereals n.e.c.': 190,
    'Mixed cereals': 191,
    'Other': 192,
    'Vegetables and melons': 200,
    'Leafy or stem vegetables': 210,
    'Artichokes': 211,
    'Asparagus': 212,
    'Cabbages': 213,
    'Cauliflowers and broccoli': 214,
    'Lettuce': 215,
    'Spinach': 216,
    'Chicory': 217,
    'Other leafy or stem vegetables n.e.c.': 219,
    'Fruit bearing vegetables': 220,
    'Cucumbers': 221,
    'Eggplants (aubergines)': 222,
    'Tomatoes': 223,
    'Watermelons': 224,
    'Cantaloupes and other melons': 225,
    'Pumpkin squash and gourds': 226,
    'Other fruit bearing vegetables n.e.c.': 227,
    'Root bulb or tuberous vegetables': 230,
    'Carrots': 231,
    'Turnips': 232,
    'Garlic': 233,
    'Onions (incl. shallots)': 234,
    'Leeks and other alliaceous vegetables': 235,
    'Other root bulb or tuberous vegetables n.e.c.': 236,
    'Mushrooms and truffles': 240,
    'Vegetables n.e.c.': 250,
    'Fruit and nuts': 300,
    'Tropical and subtropical fruits': 310,
    'Avocados': 311,
    'Bananas and plantains': 312,
    'Dates': 313,
    'Figs': 314,
    'Mangoes': 315,
    'Papayas': 316,
    'Pineapples': 317,
    'Other tropical and subtropical fruits n.e.c.': 318,
    'Citrus fruits': 320,
    'Grapefruit and pomelo': 321,
    'Lemons and Limes': 322,
    'Oranges': 323,
    'Tangerines mandarins clementines': 324,
    'Other citrus fruit n.e.c.': 325,
    'Grapes': 330,
    'Berries': 340,
    'Currants': 341,
    'Gooseberries': 342,
    'Kiwi fruit': 343,
    'Raspberries': 344,
    'Strawberries': 345,
    'Blueberries': 346,
    'Other berries': 347,
    'Pome fruits and stone fruits': 350,
    'Apples': 351,
    'Apricots': 352,
    'Cherries and sour cherries': 353,
    'Peaches and nectarines': 354,
    'Pears and quinces': 355,
    'Plums and sloes': 356,
    'Other pome fruits and stone fruits n.e.c.': 357,
    'Nuts': 360,
    'Almonds': 361,
    'Cashew nuts': 362,
    'Chestnuts': 363,
    'Hazelnuts': 364,
    'Pistachios': 365,
    'Walnuts': 366,
    'Other nuts n.e.c.': 367,
    'Other fruits': 380,
    'Oilseed crops': 400,
    'Soya beans': 410,
    'Groundnuts': 420,
    'Other temporary oilseed crops': 430,
    'Castor bean': 431,
    'Linseed': 432,
    'Mustard': 433,
    'Niger seed': 434,
    'Rapeseed': 435,
    'Safflower': 436,
    'Sesame': 437,
    'Sunflower': 438,
    'Other temporary oilseed crops n.e.c.': 439,
    'Permanent oilseed crops': 440,
    'Coconuts': 441,
    'Olives': 442,
    'Oil palms': 443,
    'Other oleaginous fruits n.e.c.': 444,
    'Root tuber crops with high starch or inulin content': 500,
    'Potatoes': 510,
    'Sweet potatoes': 520,
    'Cassava': 530,
    'Yams': 540,
    'Other roots and tubers n.e.c.': 550,
    'Beverage and spice crops': 600,
    'Beverage crops': 610,
    'Coffee': 611,
    'Tea': 612,
    'Mate': 613,
    'Cocoa': 614,
    'Other beverage crops n.e.c.': 615,
    'Spice crops': 620,
    'Chilies and peppers (capsicum spp.)': 621,
    'Anise badian and fennel': 622,
    'Other temporary spice crops n.e.c.': 623,
    'Pepper (piper spp.)': 624,
    'Nutmeg mace cardamoms': 625,
    'Cinnamon (canella)': 626,
    'Cloves': 627,
    'Ginger': 628,
    'Vanilla': 629,
    'Other permanent spice crops n.e.c.': 630,
    'Leguminous crops': 700,
    'Beans': 710,
    'Broad beans': 720,
    'Chick peas': 730,
    'Cow peas': 740,
    'Lentils': 750,
    'Lupins': 760,
    'Peas': 770,
    'Pigeon peas': 780,
    'Leguminous crops n.e.c.': 790,
    'Sugar crops': 800,
    'Sugar beet': 810,
    'Sugar cane': 820,
    'Sweet sorghum': 830,
    'Other sugar crops n.e.c.': 840,
    'Other crops and Classes': 900,
    'Grasses and other fodder crops': 910,
    'Temporary grass crops': 911,
    'Permanent grass crops': 912,
    'Fiber crops': 920,
    'Cotton': 921,
    'Jute kenaf other similar crops': 922,
    'Flax hemp and other similar products': 923,
    'Other temporary fibre crops': 924,
    'Permanent fibre crops': 925,
    'Medicinal aromatic pesticidal or similar crops': 930,
    'Temporary medicinal etc. crops': 931,
    'Permanent medicinal etc. crops': 932,
    'Rubber': 940,
    'Flower crops': 950,
    'Temporary flower crops': 951,
    'Permanent flower crops': 952,
    'Tobacco': 960,
    'Other Classes': 970,
    'Artificial Surfaces': 971,
    'Forest': 972,
    'Wetlands': 973,
    'Water bodies': 974,
    'Fallow land': 975,
    'Baren land': 976,
    'No Data Available': 977,
    'Other crops': 980,
    'Other crops temporary': 981,
    'Other crops permanent': 982,
    'Unknown crops': 998
}


id2grid = {
    0: 11,
    1: 12,
    2: 13,
    3: 21,
    4: 22,
    5: 23,
    6: 31,
    7: 32,
    8: 33
}

id2name = {v: k for k, v in CROP_ENCODING.items()}

base_dir = '/workspace/datasets/Sen4AgriNet'
labels_list = []
patch_names_list = []

lmdb_path = '/workspace/datasets/Sen4AgriNet/lmdb.db'
env = lmdb.open(str(lmdb_path), map_size=(2**40) * 2, readonly=False)


with env.begin(write=True) as txn:

    for dir_name in ['31TBF', '31TCF', '31TCG', '31TCJ', '31TCL', '31TDF', '31TDG', '31TDK', '31TDM', '31UCP', '31UDR']:
        print(dir_name)

        dir_path = os.path.join(base_dir, dir_name)
        patch_names = sorted(os.listdir(dir_path))
        for patch_name in patch_names:
            patch_path = os.path.join(dir_path, patch_name)
            dset = netCDF4.Dataset(patch_path)
            timestamps = xr.open_dataset(xr.backends.NetCDF4DataStore(dset['B02']))['B02'].values.shape[0]

            label_np = xr.open_dataset(xr.backends.NetCDF4DataStore(dset['labels']))['labels'].values
            labels = np.array(list(map(lambda x: np.split(x, 3, axis=1), np.split(label_np, 3, axis=0)))).reshape(9, 122, 122)

            for i in range(timestamps):
                bands_readout = []
                for band in ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']:
                    band_np = xr.open_dataset(xr.backends.NetCDF4DataStore(dset[band]))[band].values
                    d = int(band_np.shape[-1] / 3)
                    bands_readout.append(np.array(list(map(lambda x: np.split(x, 3, axis=1),
                                                           np.split(band_np[i], 3, axis=0)))).reshape(9, d, d))

                for grid_pos in range(9):
                    indices, counts = np.unique(labels[grid_pos], return_counts=True)
                    label_strings = [id2name[idx] for idx, count in zip(indices, counts) if idx > 0 and count > 25]
                    if len(label_strings) == 0:
                        continue

                    base_name = patch_name.split('.')[0]
                    unique_patch_name = '{}_ts{}_grid{}'.format(base_name, i + 1, id2grid[grid_pos])
                    patch_bands = list(map(lambda x: x[grid_pos], bands_readout))
                    s4a_patch = S4A_Patch(*patch_bands)
                    txn.put(unique_patch_name.encode(), s4a_patch.dumps())

                    patch_names_list.append(unique_patch_name)
                    labels_list.append(label_strings)

env.close()

df = pd.DataFrame(dict(name=patch_names_list, labels=labels_list))
df.to_parquet('/workspace/datasets/Sen4AgriNet/labels/multi_labels.parquet')
