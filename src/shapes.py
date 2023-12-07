import numpy as np


class BoundingBox:
    """
    Defines a rectangular bounding box.
    The image is indexed as a 2D numpy array, with the top left corner being (0, 0) and bottom right corner being (img.shape[0], img.shape[1]).
    """

    def __init__(self, x: int, y: int, r: int, c: int) -> None:
        """
        Params
        ------
        x, y: int
            Coordinates of the top left point of the box
        r, c: int
            Number of rows and columns of the image that fall within the bounding box

        This means that the bounding box vertices are:

        (x, y) (x, y+c)

        (x+r,y) (x+r, y+c)
        """
        self.x = x
        self.y = y
        self.r = r
        self.c = c
        self.middle = np.array([self.x + self.r / 2, self.y + self.c / 2])
        self.area = self.r * self.c

    def distance(self, other):
        """
        Compute the distance of the midpoint of the two bounding boxes
        """
        return np.linalg.norm(self.middle - other.middle)

    def merge(self, other):
        """
        Merges the two bounding boxes
        """
        x = min(self.x, other.x)
        y = min(self, y, other.y)
        r = max(self.x + self.r, other.x + other.r) - x
        c = max(self.y + self.c, other.y + other.c) - y
        return BoundingBox(x, y, r, c)

    def crop(self, img):
        return img[self.x : self.x + self.r, self.y : self.y + self.c]

    def get_corner(self) -> tuple:
        return self.x, self.y

    def get_width(self) -> int:
        return self.c

    def get_height(self) -> int:
        return self.r

    def get_center(self) -> np.ndarray:
        return self.middle
