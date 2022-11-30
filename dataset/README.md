
# HUMANISE Synthesis

Following the next steps to synthesize your own data.

## 1. Dataset Preparation

- [AMASS](https://amass.is.tue.mpg.de/) dataset
    - In our implementation, we only use the subset 'ACCAD', 'BMLmovi', 'BMLrub', 'CMU', 'EyesJapanDataset', 'MPIHDM05', 'KIT' in AMASS.
    - You can silghtly change the code to use more other subsets. Please refer to `./dataset/babel_process.py` for more details. Remember to change the folder path.

- [BABEL](https://babel.is.tue.mpg.de/) dataset
    - In our implementation, we only use the annotation in `train.json` and `val.json` in BABEL.

- [ScanNet V2](http://www.scan-net.org/) dataset
    - In our implementation, we use the scenes with scan_id from `scene0000_00` to `scene0706_00`.

- [Scan2Cad](https://github.com/skanti/Scan2CAD) dataset
    - In our implementation, we use the file `full_annotations.json`

- [ReferIt3D](https://referit3d.github.io/) dataset
    - In our implementation, we use `sr3d` annotation, which contains referential language descriptions for the objects.

## 2. Extract Motion Segments from BABEL

Run the following command to extract the motion segments.
    
```bash
python dataset/babel_process.py --action "$ACTION" # e.g., dataset/python babel_process.py --action "stand up"
```

- In our original implementation, we only consider four actions, i.e., sit, stand up, walk, and lie.
- This script will first generate a `'./dataset/action_segment/{$ACTION}.json'` file that contains satisfied motions' information from BABEL annotation. You can modify Line83 in `./dataset/babel_process.py` to extract longer motion sequences.
- Then, the script will extract the motion segments according to the BABEL annotations and save motions into the target folder which is specified by `motion_seg_folder` at Line18 in `dataset/babel_process.py`.


You can directly download our pre-processed motion segments from the [link](https://docs.google.com/forms/d/e/1FAIpQLSfzhj2wrRLqAXFVOTn8K5NDN-J_5HueRTohMAlayqBuPPWA1w/viewform?usp=sf_link). (The pre-processed motion segments are in the `pure_motion` folder in the shared drive folder.)

## 3. Align Motion Segments with ScanNet Scenes

Remember to change the folder path in `utils/configuration.py`.

Run the following command to align the extracted motion segments with scannet scenes.

```bash
python dataset/align_motion.py --anno_path=$SAVE_PATH --action=$ACTION --max_s=1 --use_lang --sort_scene --save --visualize --rendering
# e.g., python dataset/align_motion.py --anno_path=./align_data/ --action="sit" --max_s=1 --use_lang --sort_scene --save --visualize --rendering
```

- `--anno_path`: the path to save aligned results
- `--action`: specify the action type
- `--max_s`: sample `max_s` motion(s) for each interacting object and motion pair
- `--use_lang`: use `sr3d` language annotations to annotate the aligned human-scene interaction
- `--sort_scene`: process the scenes in order
- `--save`: save the aligned results into `pkl` files
- `--visualize`: visualize the aligned results on screen
- `--rendering`: render the aligned results into videos and gifs (time-consuming)


You can directly download our pre-aligned human-scene interactions from the [link](https://docs.google.com/forms/d/e/1FAIpQLSfzhj2wrRLqAXFVOTn8K5NDN-J_5HueRTohMAlayqBuPPWA1w/viewform?usp=sf_link).

## File Format

The `babel_process.py` in step2 will extract a motion segments from original AMASS motion seuqences into `motion.pkl`. The folder structure and file format are as follows:

- folder structure

    ```bash
    -| $motion_seg_folder
    ---| $ACTION
    -----| ${babel_sid}_${seg_id}
    -------| motion.pkl
    ```

- `motion.pkl` format
    - This file store a tuple.

    ```python
    (
        gender,         # str, nerutal
        trans,          # np.ndarray, <L, 3>
        orient,         # np.ndarray, <L, 3>
        betas,          # np.ndarray, <16>
        body_pose,      # np.ndarray, <L, 63>
        hand_pose,      # np.ndarray, <L, 90>
        jaw_pose,       # np.ndarray, <L, 3>
        eye_pose,       # np.ndarray, <L, 6>
        joints,         # np.ndarray, <L, 127, 3>
    )
    ```

---

The `align_motion.py` in step3 will save the aligned motion into `anno.pkl` files. The folder structure and file format are as follows:

- folder structure

    ```bash
    -| $SAVE_PATH
    ---| $ACTION
    -----| ${scene_id}_${babel_sid}_${seg_id}
    -------| anno.pkl
    ```

- `anno.pkl` format
    - This file stores a list, in which each item is a aligned result. Each scene and motion segment pair may have more than one results, as there are more than one interacting objects in the scene.

    ```python
    [
        {
            'action': str, # action type
            'motion': str, # motion segment id, i.e., ${babel_sid}_${seg_id}
            'scene':  str, # scene id
            'scene_translation': np.ndarray, # scene translation array used for normalizing scene to coordinate origin, <3>

            'translation': np.ndarray, # translation array for motion sequence, <3>
            'rotation': float, # rotation angle for motion sequence
            'utterance': str, # language description
            'object_id': int, # object id in the scene (scannet annotation)
            'object_label': str, # semantic label of interacting object
            'object_semantic_label': int, # semantic label id of interacting object, see `dataset/data/label_mapping.json`
        },
        ...
    ]
    ```
