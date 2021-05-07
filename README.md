


# Incidental Lung Nodule Project Documentation


## Nodule Detection

### Data Preparation
#### Workflow
> Create ***details.xlsx***/***checklist.xlsx*** --> Pull **dicom** images --> Get raw/normalized data saved in **.npz** format --> Annotate the **central slices**  --> Generate location label file ***pos_label_norm.csv*** --> Preprocess data by **lung segmentation** and **cropping**.


 #### 1. Create details.xlsx/checklist.xlsx
- ***details.xlsx*** is used to save the main information (patient-level) for this dataset, it contains the **patient Idx (pstr)**, **MRN**, annotated nodules, **pathology label**, has annotation flag, date of surgery, assign to. **Bold** items are information must be given at the beginning. Sample ***details.xlsx***:
	![details](https://www.dropbox.com/s/119y3kk5xu25snn/Details_line.png?raw=1)
	
- ***checklist_{ext}.xlsx*** can be generated from the ***details.xlsx*** and it is used to control the data (nodule-level) we want to precessed at the round, where **{ext}** is an extra string to identify this checklist. Each checklist contains the **patient Idx (pstr)**, **MRN**,  **date (dstr)**, **Series**, **z**, nodule Index, main nodule, PET, position, szie, compare with. **date (dstr)**, **Series**, **z**, nodule Index, main nodule, PET, position, szie, compare with, can be obained from checking the report on the ***APEC*** system.  **z** is not required if we don't want annotate this nodule (e.g. in inference mode). **Series** is necessary for external exams (CT obtained outside methodist).  Sample ***checklist.xlsx***:
	![enter image description here](https://www.dropbox.com/s/1vbkvbtswlim6ek/Checklist_line.png?raw=1)
	
#### 2. Pull dicom images [*On Windows*]
In order to pull the specified dicom images, we need to first generate a list of identifiers to specify what data to pull from the database, then log into the computer that can access the database t run the batch code to pull the dicom data automatically. 
##### Generate  t of i to get identifiers.csv
```python data_utils.py get_id -s={SAVE_DIR} -c={CK_DIR} -p={PRE_DIR} -ext={EXTRA_STR}```
- `get_id` : Task is to get identifiers.csv 
- `--save_dir`(`-s`): [***str***] Save directory
- `--ck_dir`(`-c`): [***str***] Checklist directory
- `--pre_dir`(`-p`): [***str***, ***optional***] Directory where some data is preprocessed/pre-downloaded
- `--extra_str`(`-ext`): [***str***, ***optional***] Extra string for the checklist

If `--pre_dir` is given, there will be a `move_data.csv` indicating the data to be moved. Run the following command to move those pre-downloaded data:
```python data_utils.py move_data -s={SAVE_DIR}```
- `move_data`: Task is to move pre-downloaded dicom data to save directory
- `--save_dir`(`-s`): [***str***] Save directory

Example:
If the checklist file is located at `Z:\Methodist_incidental_Kim\checklist_Ben.xlsx`, then `-c=Z:\Methodist_incidental_Kim`, `-ext=Ben`, 

##### Automatically download dicom data
- Set `ROOTFOLDER` in `pull_dicom_data.bat` file by: `set ROOTFOLDER={SAVE_DIR}`, where `{SAVE_DIR}`is where `identifiers.csv` located. *Note that `{SAVE_DIR}` must ends with a `"\"` here*.
- Run ```.\pull_dicom_data.bat``` in terminal.

In the end, all dicom data will be saved in `{SAVE_DIR}`, *Note that all series are saved, but normally we only need Lung Window (PET or not PET)*. The data strucuture is shown as followed:
```
SAVE_DIR
|   identifiers.csv
|   move_data.csv
└── Lung_patient{pstr0}-{MRN0}_{date0}
|    |   01DATAINFO.txt
|    |   uids.csv
|    └── {date0}_CT_data
|         |   slice0.dcm
|         |   ... 
└── Lung_patient{pstr0}-{MRN0}_{date1}
│    |   ... 
```
#### 3. Get raw/normalized data saved in **.npz** format [*On Windows*]
This step is to de-identify the data by saving it in .npz format.
```python data_utils.py get_npz -r={ROOT_DIR} -c={CK_DIR} -n={NORMALIZE} -ext={EXTRA_STR}```

- `get_npz` : Task is to save data in .npz format
- `--root_dir`(`-r`): [***str***] Root directory that stores the downloaded dicom data
- `--save_dir`(`-s`): [***str***] Save directory that stores the npz data
- `--ck_dir`(`-c`): [***str***] Checklist directory
- `--normalize`(`-n`): [***bool***] Whether to normalize data 
- `--pre_dir`(`-p`): [***str***, ***optional***] Directory where some data is preprocessed and saved in .npz format
- `--extra_str`(`-ext`): [***str***, ***optional***] Extra string for the checklist

Same as in step 2,  If `--pre_dir` is given. Run the following command to move those pre-processed data:  
`python data_utils.py move_data -s={SAVE_DIR}`

In the end, all npz data will be saved in `{SAVE_DIR}\normalized` or `{SAVE_DIR}\raw``, *Note that only the corresponding series are saved. The data strucuture is shown as followed:
```
SAVE_DIR
└── normalized/raw
|    |   CTinfo.npz
|    |   move_data.csv
|    |   log
|    └── Lung_patient{pstr0}-{MRN0}
|         |   {MRN0}-{date0}.npz
|         |   {MRN0}-{date1}.npz
│         |   ... 
|    └── Lung_patient{pstr10}-{MRN10}
|    |   ... 
```

#### 4. Annotate the central slices [*On Linux*]
##### Extract the central slices from the de-identified data.
```python prepare.py extract -r={ROOT_DIR} -s={SAVE_DIR} -p={CK_PATH} -n={NORMALIZE}```
- `extract` : Task is to extract central slices
- `--root_dir`(`-r`): [***str***] Root directory that stores the npz data
- `--save_dir`(`-s`): [***str***] Save directory that stores the central slices, they will be saved in `SAVE_DIR/central_slices_{norm/raw}`
- `--ck_path`(`-p`): [***str***] Checklist file full path
- `--normalize`(`-n`): [***bool***] Whether to normalize data 

##### Mannually annotation based on the the central slices and checklist.xlsx
Go to open-source website [make sense](https://www.makesense.ai/) to annotate all the central slices by hand. Save the annotations as a single csv file (e.g. labels_my-project-name_2021-04-20-11-13-39.csv). Recommended to annotate the normalized data.

##### Create GT label (Change the format of annotations)
```python convert -r={ROOT_DIR} -s={SAVE_DIR} -a={ANNOT_FILE} -n={NORMALIZE}```
- `convert` : Task is to convert annotation from make sense output to normalized ground truth `pos_label_norm.csv`
- `--root_dir`(`-r`): [***str***] Root directory that stores the npz data
- `--save_dir`(`-s`): [***str***] Save directory that contains the make sense annotation file and will be used to save GT label
- `--annot_file`(`-a`): [***str***] Name of the annotation file from make sense
- `--normalize`(`-n`): [***bool***] Whether the annotated data is normalized or not 

#### 5.  Preprocess data by lung segmentation and cropping
```python prepare.py prep_methodist -s={SAVE_DIR} -r={ROOT_DIR} -m={MASK} -c={CROP}```
- `prep_methodist`: Task is to do preprocessing for methodist data
- `--root_dir`(`-r`): [***str***] Root directory that stores the npz data
- `--save_dir`(`-s`): [***str***] Save directory that is used to save preprocessed data
- `--mask`(`-m`): [***bool***] Apply unsupervised lung mask in preprocessing
- `--crop`(`-c`): [***bool***] Crop masked images in preprocessing

### Run detection code
#### Train
```python detect.py -d=methodistFull --test=False --gpu="2,3" --start-epoch=0```
#### Test
```python detect.py -d=methodistFull --test=True --gpu="2,3" --resume="../detector_ben/results/res18-20201202-112441/026.ckpt"```
#### Inference
1. Set `POS_LABEL_FILE = None` in `methodistFull.py` file.
2. ```python detect.py -d=methodistFull --test=True --gpu="2,3" --resume="../detector_ben/results/res18-20201202-112441/026.ckpt"```
#### Parameters

- `--datasource`(`-d`): [***str***] Datasource (options: [luna, methodist])
- `--model`(`-m`): [***str***] Model to be used  (default: 3D Resnet-18)
- `--epochs`(`-e`): [***int***] Number of total epochs to run
- `--start-epoch`: [***int***] Manually set start epoch number
- `--batch-size`(`-b`): [***int***] Batch size
- `--learning-rate`(`-lr`): [***float***] Initial learning rate
- `--resume`(`-re`): [***str***] Path to latest checkpoint (default: none)
- `--save_dir`(`-s`): [***str***] Save directory that is used to save results
- `--test`(`-t`): [***bool***] True if in test mode, otherwise in train or inference mode
- `--inference`(`-i`): [***bool***] True if in inference mode, otherwise in train or test mode
- `--gpu`: [***str***] GPUs to be used
- `--mask`(`-m`): [***bool***] Apply unsupervised lung mask in preprocessing
- `--crop`(`-c`): [***bool***] Crop masked images in preprocessing




