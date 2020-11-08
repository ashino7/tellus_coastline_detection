1. Environment:
  - python>=3.6
  - library
    - numpy==1.16.3

2. Usage:
  - In the command line, execute
    ```
    $ python evaluate.py --prediction-file path/to/prediction --annotation-file path/to/annotation
    ```
    and you can get the "ExtractError" score of the prediction for the given ground truth(annotation).

3. Notes:
  - The format of the prediction file:
    - File Name: ***.json (*** = whatever name you like(e.g. predictions))
    - Description:
      - image_file_0: [[x1, y1],...]
      - image_file_1: [[x1, y1],...]
      ...
    - The origin of the coordinate system in the image is top left.
    - [x1, y1] corresponds to [x-coordinate(horizontal), y-coordinate(vertical)] in the image.
    - For each image, the predicted coastline is expressed as the set of the coordinates as in the description.
    - Please also refer to "sample_submit.json".
  - The format of the annotation file
    - File Name: ***.json (*** = whatever name you like(e.g. annotations))
    - Description:
      - image_flie_0:
        - coastline_points: [[x1, y1], ...]
        - validate_lines: [[[x1, y1], [x2, y2]], ...]
      - image_file_1:
        - coastline_points: [[x1, y1], ...]
        - validate_lines: [[[x1, y1], [x2, y2]], ...]
      ...
    - The origin of the coordinate system in the image is top left.
    - In "coastline_points", [x1, y1] corresponds to [x-coordinate(horizontal), y-coordinate(vertical)] in the image.
    - For each image, the coastline is expressed as the set of the coordinates as in the description.
    - In "validate_lines", [x1, y1] and [x2, y2] correspond to the end points of the validate line(line segment).