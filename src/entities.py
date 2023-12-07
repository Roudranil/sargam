import numpy as np
import shapes
import utils
import cv2


class Sheet:
    """
    Class to store every sheet of the input
    """

    def __init__(self, img) -> None:
        # start = time.time()
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.img = img
        # print_log("[IMAGE IMPORTED]", start)
        # start = time.time()

        # image properties, lengths and indices
        self.line_spacing, self.line_width = utils.get_reference_lengths(self.img)
        # print_log("[COMPUTED REF LENGTHS]", start)
        # start = time.time()

        self.staff_length = (5 * self.line_width) + (4 * self.line_spacing)
        self.all_staff_row_indices = utils.find_staffline_rows(
            self.img, self.line_width, self.line_spacing
        )
        # print_log("[COMPUTED STAFF ROWS]", start)
        # print(len(self.all_staff_row_indices))
        # start = time.time()
        self.all_staff_col_indices = utils.find_staffline_columns(
            self.img, self.all_staff_row_indices, self.staff_length
        )
        # print_log("[COMPUTED STAFF COLUMNS]", start)
        # print(len(self.all_staff_col_indices))
        # start = time.time()
        self.half_sep = utils.get_staff_separation(
            self.all_staff_row_indices, self.staff_length
        )
        # print_log("[COMPUTED STAFF SEPARATION]", start)
        # start = time.time()
        chunk_indices = utils.get_chunks(self.img, self.staff_length, self.half_sep)
        self.all_chunk_indices = sorted(chunk_indices, key=lambda x: x[1])
        # print_log("[COMPUTED CHUNK INDICES]", start)
        # print(self.all_chunk_indices)
        # start = time.time()
        self.chunks = self._get_chunks_list()
        # print_log("[COMPUTED CHUNK LIST]", start)
        # print(len(self.chunks))
        # start = time.time()
        self.staff_boxes = self._get_staves_list()
        # print_log("[COMPUTED STAFF LIST]", start)

    def _get_chunks_list(self):
        chunks = []
        row_extremes = [[r[0][0], r[-1][-1]] for r in self.all_staff_row_indices]
        i = 0
        for chunk_start, chunk_end in self.all_chunk_indices:
            rows_in_chunk = []
            iter_range = range(i, len(row_extremes))
            for i in iter_range:
                row_start, row_end = row_extremes[i]
                if (
                    chunk_start - self.line_spacing <= row_start
                    and row_end <= chunk_end + self.line_spacing
                ):
                    rows_in_chunk.append(i)
                else:
                    break
            chunks.append(rows_in_chunk)
        return chunks

    def _get_staves_list(self):
        staff_boxes = []
        for row_idx, col_idx in zip(
            self.all_staff_row_indices, self.all_staff_col_indices
        ):
            x = row_idx[0][0]  # get the first row of the first line, top left corner x
            y = col_idx[0]  # get the first column extreme, top left corner y
            r = row_idx[-1][-1] - x
            c = col_idx[-1] - y
            # adding the half sep as padding
            staff_box = shapes.BoundingBox(
                max(0, x - self.half_sep),
                y,
                min(x + r + 2 * self.half_sep, self.img.shape[0] - 1) - x,
                c,
            )
            staff_boxes.append(staff_box)
        return staff_boxes
