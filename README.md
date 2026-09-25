# SEAMAN FishID `.ID` exporter

Python utility to convert SEAMAN FishID `Training_Data` binary files into acoustic images, NumPy arrays and per-channel metadata.

This repository packages the processing used for the SEAMAN FishID dataset as a reusable command-line program. Dataset-specific campaign paths were removed; the input and output directories are supplied when the program is run.

## What the program does

For every `Seaman_Fish_ID_Training_Data_*.ID` file found in the input directory, the program:

1. reads the FishID binary headers and acoustic data products;
2. obtains the bottom position, preferably from `Depth_In_Samples` when enough valid ping values are available;
3. uses `Data3` (`S5`, DB data without TVG) to generate the acoustic JPG image;
4. removes the region below the detected bottom and aligns the retained water-column section;
5. exports the available acoustic products (`S2`, `S3`, `S5` and `S7`) as `.npy` files;
6. writes one JSON file per channel with acquisition, acoustic and image-geometry metadata.

A dedicated Split Beam reader is included for the supported FishID layout. It exports CH1-CH4 and the virtual S7 channel separately.

## Requirements

- Python 3.10 or newer is recommended.
- Packages listed in `requirements.txt`.

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Input

`--input-dir` must point to one maneuver directory containing one or more files named:

```text
Seaman_Fish_ID_Training_Data_*.ID
```

The directory may also contain:

```text
Seaman_Fish_ID_File_Features_Maneouver_*.ID
```

The Maneouver file is optional. When present, the program reads it once and includes its relevant metadata in the output JSON files. If it is absent, the acoustic files are still processed.

Example input directory:

```text
my_maneuver/
├── Seaman_Fish_ID_File_Features_Maneouver_1.ID
├── Seaman_Fish_ID_Training_Data_1.ID
├── Seaman_Fish_ID_Training_Data_2.ID
└── Seaman_Fish_ID_Training_Data_3.ID
```

The binary reader was developed for the FishID layouts represented in the SEAMAN dataset used by this project. A file produced by a different firmware/layout may require an extension of the reader.

## Basic use

```bash
python seaman_id_to_yolo.py \
    --input-dir /path/to/my_maneuver \
    --output-dir /path/to/output \
    --prefix campaign_01
```

`--prefix` is optional. If omitted, the name of the input directory is used.

Before processing a complete dataset, a short test can be run with:

```bash
python seaman_id_to_yolo.py \
    --input-dir /path/to/my_maneuver \
    --output-dir /path/to/test_output \
    --prefix test \
    --max-blocks 20
```

Run:

```bash
python seaman_id_to_yolo.py --help
```

for all available options.

## Main command-line options

| Option | Meaning |
|---|---|
| `--input-dir` | Directory containing the FishID `.ID` files. Required. |
| `--output-dir` | Directory where JPG, NPY and JSON files are written. Required. |
| `--prefix` | Prefix added to every output filename. Defaults to the input-directory name. |
| `--global-output-dir` | Optional second directory receiving a copy of every generated file. |
| `--max-blocks` | Limit the number of pings/blocks per Training Data file. Intended for testing. |
| `--bottom-method A|B` | Fallback bottom detector when `Depth_In_Samples` cannot be used. Default: `A`. |
| `--no-depth-in-samples` | Forces bottom estimation instead of using `Depth_In_Samples`. |

## Output: general processing path

For a Training Data file with stem:

```text
Seaman_Fish_ID_Training_Data_1
```

and prefix `campaign_01`, typical files for CH1 are:

```text
campaign_01_Seaman_Fish_ID_Training_Data_1_CH1.jpg
campaign_01_Seaman_Fish_ID_Training_Data_1_CH1_Data1_S2_FILTERED.npy
campaign_01_Seaman_Fish_ID_Training_Data_1_CH1_Data2_S3_CORRELATOR_DETECTOR.npy
campaign_01_Seaman_Fish_ID_Training_Data_1_CH1_Data3_S5_DB_WITHOUT_TVG.npy
campaign_01_Seaman_Fish_ID_Training_Data_1_CH1_Data4_S7_DB.npy     # when available
campaign_01_Seaman_Fish_ID_Training_Data_1_CH1.json
```

Equivalent files are generated for the remaining detected channels.

### JPG

The final JPG is generated from `Data3 = S5_DB_WITHOUT_TVG`.

The current default processing is the one used for the dataset generation:

- `Depth_In_Samples` is used for the bottom when at least 50% of the pings provide a valid value;
- otherwise the configured bottom detector is used;
- the region below the bottom is excluded;
- the image contains the available water column above the bottom;
- the original contrast/palette processing is preserved.

### NPY files

In the general processing path, S2/S3/S5/S7 are extracted from the same physical section represented in the final JPG. The matrices are then resampled to the saved JPG pixel dimensions and vertically oriented to match the image.

The saved layout is:

```text
axis 0 -> pings / horizontal JPG coordinate
axis 1 -> vertical samples / vertical JPG coordinate
```

Therefore, in this path:

```text
npy.shape[0] == JPG width
npy.shape[1] == JPG height
```

S3 can be complex depending on the FishID acquisition mode/layout. Complex values are preserved in the `.npy` output.

### JSON

The per-channel JSON contains selected metadata useful for acoustic interpretation, including acquisition parameters, channel information, frequency-related fields, depth/range information, bottom-alignment information and image geometry.

## Split Beam output

For the supported dedicated Split Beam layout, the program exports the four real channels with:

```text
*_CH1.jpg ... *_CH4.jpg
*_CHx_Data1_S2_FILTERED.npy
*_CHx_Data2_S3_CORRELATOR_DETECTOR.npy
*_CHx_Data3_S5_DB_WITHOUT_TVG.npy
*_CHx_Data4_S7_DB_VIRTUAL.npy
*_CHx.json
```

It also exports the virtual channel:

```text
*_VIRTUAL_S7.jpg
*_VIRTUAL_Data3_S5_DB_WITHOUT_TVG.npy
*_VIRTUAL_Data4_S7_DB.npy
*_VIRTUAL_S7.json
```

For Split Beam, S7 belongs to the virtual channel. `*_CHx_Data4_S7_DB_VIRTUAL.npy` is therefore the same virtual S7 information saved with a per-channel filename so that downstream datasets can keep a homogeneous Data1-Data4 naming convention.

### Important Split Beam layout note

The dedicated Split Beam branch preserves the same physical section above the bottom, but its NPY storage is not pixel-resampled in exactly the same way as the general processing branch. It saves the cropped arrays as `(pings, samples)`. This behavior is intentionally retained here to reproduce the dataset-generation code rather than silently changing existing results.

## Processing defaults used for reproducibility

The script keeps the original dataset-generation defaults, including:

```text
image source                 Data3 / S5_DB_WITHOUT_TVG
prefer Depth_In_Samples      yes
minimum valid depth fraction 0.50
fallback bottom method       A
min range ratio              0.55
max range ratio              0.90
profile smoothing            9
maximum bottom jump          50 samples
bottom smoothing window      21
reference height             60 m
output DPI                    200
```

Changing these values changes the generated representation. For reproducibility, leave them at their defaults unless a different processing experiment is intended.

## Optional global output directory

To collect all generated files from several maneuver directories in one location, use:

```bash
python seaman_id_to_yolo.py \
    --input-dir /data/maneuver_01 \
    --output-dir ./maneuver_01_output \
    --prefix maneuver_01 \
    --global-output-dir ./all_generated_files
```

Run the program once for each maneuver/campaign, using a different prefix.

## What this program does not do

This script performs `.ID` decoding and generation of channel images/matrices/metadata. It does not perform YOLO annotation, object detection, manual relabeling or extraction of object/fish bounding-box crops.

## Reproducibility note

The two original notebooks used for the first and second dataset batches contained the same processing algorithm. Their differences were the hard-coded campaign lists and the name of the aggregate output directory. This public version is based on that common algorithm and replaces those dataset-specific paths with command-line arguments.

## License

No software license is included yet. Add the license selected by the project owners before publishing the repository if reuse or redistribution by third parties is intended.

## Notebook para múltiples campañas

El archivo `SEAMAN_FishID_generar_YOLO_MULTIPLES_CAMPANIAS.ipynb` permite procesar varias carpetas en una sola ejecución.

Cada campaña se configura como:

```python
{
    "input_dir": "/ruta/a/Seaman_Fish_ID_Maneouver_1",
    "prefijo": "campania_01",
    "output_dir": "./campania_01/imagenes_yolo",
}
```

La lista incluida en la notebook es **solo un ejemplo**. Debe reemplazarse por las rutas reales de cada usuario.

La notebook incluye una verificación previa de carpetas y archivos antes de comenzar el procesamiento.
