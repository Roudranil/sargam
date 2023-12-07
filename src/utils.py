import numpy as np
import cv2
from tqdm import tqdm
from copy import deepcopy
from shapes import BoundingBox
from collections import Counter


def match(
    img: np.ndarray,
    templates: np.ndarray,
    start_percent: float,
    stop_percent: float,
    threshold: float,
):
    best_location_count = -1
    best_locations = []
    best_scale = 1
    loop = tqdm([i / 100.0 for i in range(start_percent, stop_percent + 1)])
    for scale in loop:
        loop.set_description_str(f"Looking for template scaled to {scale}%%")
        locations = []
        location_count = 0

        for i, template in enumerate(templates):
            if (scale * template.shape[0] > img.shape[0]) or (
                scale * template.shape[1] > img.shape[1]
            ):
                continue

            template = cv2.resize(
                template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            results = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            results = np.argwhere(results >= threshold)
            location_count += len(results)
            locations += list(results)
            loop.set_postfix_str(f"{len(results)} hits for template {i}")

        if location_count > best_location_count:
            best_location_count = location_count
            best_locations = locations
            best_scale = scale
        else:
            pass

    return best_locations, best_scale


def find_staffline_rows(img, line_width, line_spacing):
    """
    Calculates the indices of the rows that have staff lines, with a margin of error, because cropping is done by padding half seps

    Parameters
    ----------
    img: np.ndarray
        The grayscale image as a numpy array (assuming the only values are 0 and 255)
    line_width: int
        Number of rows spanning one staff line
    line_spacing: int
        Number of rows spanning one gap between two consecutive staff lines
    """
    # optimisation result:
    # for 1 page
    # 24.6s -> 0.1s
    # approx 99.6% reduction in time
    # approx 246x speedup
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)

    # Determine number of black pixels in each row
    # before that convert the image to a binary one again
    binary_image = np.where(img == 0, 1, 0)
    row_black_pixel_sum = np.sum(binary_image, axis=1)

    all_staff_row_indices = []
    num_stafflines = 5
    threshold = 0.3

    # Note that previous code considered ALL rows as candidates to have a staff line
    # Here we take a strict set of candidates which have a certain required number of black pixels
    # Problem with this is that this will one of the rows of the staff line
    # So take a lenient set of candidates which is 1 line width before and after this
    # Looping will ensure all the necessary rows are picked
    strict_candidates = np.where(row_black_pixel_sum > 0.7 * num_cols)[0]
    candidates = np.sort(
        np.unique(
            np.concatenate(
                [
                    strict_candidates + k
                    for k in range(-line_width, line_width + 1, 2 * line_width)
                ]
            )
        ),
        kind="quicksort",
    )

    # Find stafflines by finding sum of rows that occur according to
    # staffline width and staffline space which contain as many black pixels
    # as a thresholded value (based of width of page)
    #
    # Filter out using condition that all lines in staff
    # should be above a threshold of black pixels
    for current_row in candidates:
        indices = [
            list(range(j, j + line_width))
            for j in range(
                current_row,
                current_row + (num_stafflines - 1) * (line_width + line_spacing) + 1,
                line_width + line_spacing,
            )
        ]
        staff_lines = row_black_pixel_sum[indices]

        for line in staff_lines:
            if sum(line) / line_width < threshold * num_cols:
                current_row += 1
                break
        else:
            staff_row_indices = indices
            all_staff_row_indices.append(staff_row_indices)

    extremes = np.array([[_[0][0], _[-1][-1]] for _ in all_staff_row_indices])
    extremes_flat = np.ravel(extremes)
    diff = np.diff(extremes_flat)
    coords = (np.where(diff < 0)[0] + 1) // 2
    # for c in coords:
    #     a = np.array(all_staff_row_indices[c - 1])
    #     b = np.array(all_staff_row_indices[c])
    #     all_staff_row_indices[c - 1] = np.unique(
    #         np.concatenate([a, b]), axis=0
    #     ).tolist()

    for c in coords[::-1]:
        _ = all_staff_row_indices.pop(c)

    return all_staff_row_indices


def find_staffline_columns(img, all_staff_row_indices, staff_length):
    binary_image = np.where(img == 0, 1, 0)
    num_cols = img.shape[1]
    staff_length = staff_length
    all_staff_col_indices = []
    for row_idx in all_staff_row_indices:
        staff = binary_image[row_idx[0][0] : row_idx[-1][-1], :]
        current_beginning = []
        current_ending = []
        for col in range(num_cols // 2):
            left = staff[:, col]
            right = staff[:, -col]
            if np.sum(left) == 0:
                current_beginning.append(col)
            if np.sum(right) == 0:
                current_ending.append(num_cols - col)
        if current_beginning and current_ending:
            all_staff_col_indices.append(
                [np.max(current_beginning), np.min(current_ending)]
            )

    return all_staff_col_indices


def remove_stafflines(img, all_staffline_vertical_indices):
    no_staff_img = deepcopy(img)
    for staff in all_staffline_vertical_indices:
        for line in staff:
            for row in line:
                # Remove top and bottom line to be sure
                no_staff_img[row - 1, :] = 255
                no_staff_img[row, :] = 255
                no_staff_img[row + 1, :] = 255

    return no_staff_img


def locate_templates(img, templates, start, stop, threshold):
    locations, scale = match(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        img_locations.append(
            [BoundingBox(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])]
        )
    return img_locations


def merge_boxes(boxes, threshold):
    filtered_boxes = []
    while len(boxes) > 0:
        r = boxes.pop(0)
        boxes.sort(key=lambda box: box.distance(r))
        merged = True
        while merged:
            merged = False
            i = 0
            for _ in range(len(boxes)):
                if r.overlap(boxes[i]) > threshold or boxes[i].overlap(r) > threshold:
                    r = r.merge(boxes.pop(i))
                    merged = True
                elif boxes[i].distance(r) > r.w / 2 + boxes[i].w / 2:
                    break
                else:
                    i += 1
        filtered_boxes.append(r)
    return filtered_boxes


def get_reference_lengths(img: np.ndarray) -> list:
    """
    calculates the width (number of rows) taken for one staff line and the number of rows between two stafflines.

    Parameters
    ----------
    img: np.ndarray
        The grayscale image as a numpy array (assuming the only values are 0 and 255)

    Returns
    -------
    retval_spacing_width: list
        line_spacing and line_width
    """
    # optimisation result:
    # for 1 page
    # 3s -> 0.9s
    # 70% reduction in time
    # 3.33x speedup
    retval_spacing_width = []
    for pixel_value in [255, 0]:
        # create a boolean mask for the pixel value and flatten it
        pixels = np.where(img == pixel_value, 1, 0)
        flat = np.ravel(pixels, "F")
        # this flattening of the image is done in the original code
        # by extending the big runs list with the runs per column

        # calculate where each run starts and ends
        # consider edge cases
        # calculate their lengths
        run_starts = np.where(np.diff(flat) == 1)[0] + 1
        run_ends = np.where(np.diff(flat) == -1)[0]
        if flat[0] == 1:
            run_starts = np.insert(run_starts, 0, 0)
        if flat[-1] == 1:
            run_ends = np.append(run_ends, len(flat) - 1)
        run_lengths = run_ends - run_starts + 1

        # Get the most common count of the lengths
        run_counter = Counter(run_lengths)
        retval_spacing_width.append(run_counter.most_common(1)[0][0])

    return retval_spacing_width


def get_staff_separation(all_staff_row_indices, staff_length):
    if len(all_staff_row_indices) <= 1:
        return staff_length
    seps = []
    for curr_set, next_set in zip(all_staff_row_indices, all_staff_row_indices[1:]):
        gap = next_set[0][0] - curr_set[-1][-1]
        seps.append(gap)
    min_sep = min(seps)
    half_sep = min_sep // 2
    return half_sep


def get_chunks(img, staff_length, half_sep):
    min_chunk_length = 2 * (staff_length)
    num_rows = img.shape[0]
    num_cols = img.shape[1]
    binary_image = (img == 0).astype(int)
    chunks = []
    for col in range(num_cols):
        col_data = binary_image[:, col]
        run_starts = np.where(np.diff(col_data) == 1)[0] + 1
        run_ends = np.where(np.diff(col_data) == -1)[0]
        if col_data[0] == 1:
            run_starts = np.insert(run_starts, 0, 0)
        if col_data[-1] == 1:
            run_ends = np.append(run_ends, num_rows - 1)
        run_lengths = run_ends - run_starts + 1
        valid_runs = np.where(run_lengths >= min_chunk_length)[0]
        for idx in valid_runs:
            chunks.append([run_starts[idx] - half_sep, run_ends[idx] + half_sep])

    return list(set(map(tuple, chunks)))
