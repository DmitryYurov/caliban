
## Dataset description

The dataset is used in [Solving the robot-world hand-eye(s) calibration problem with iterative methods](https://link.springer.com/article/10.1007/s00138-017-0841-7) (also see [the preprint on arXiv](https://arxiv.org/abs/1907.12425)). The original data can be found [here](https://agdatacommons.nal.usda.gov/articles/dataset/Data_from_Solving_the_Robot-World_Hand-Eye_s_Calibration_Problem_with_Iterative_Methods/24667896). Note that the change of chessboard orientation is too big for some of OpenCV detection methods to keep the same ordering of the points for all the images. It is recommended to visually inspect if the order of the corners is the same at each image wrt the orientation.

### Files and Format

- `calibration_object.txt:` contains the information about the dimensions of the object
- `robot_cali.txt:` the number of robot poses and base-to-hand transform for each pose with translation values in mm. Poses are aligned with the image index.

