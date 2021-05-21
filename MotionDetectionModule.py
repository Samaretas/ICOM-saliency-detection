import numpy as np
from cv2 import cv2
from scipy.ndimage.filters import gaussian_filter

class MotionDetection:

    def __init__(self):
        self.block_size = 5
        self.large_search_pattern_offsets = [
            (0, 0),
            (2, 0),
            (1, 1),
            (0, 2),
            (-1, 1),
            (-2, 0),
            (-1, -1),
            (0, -2),
            (1, -1)
        ]
        self.small_search_pattern_offsets = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1)
        ]
        self.max_offset_l = 2
        self.max_offset_s = 1
        self.previous_motion = None

    def block_difference(self, ba, bb):
        return np.sum(np.abs(ba-bb))

    def diamond_search_motion_estimation(self, prev_frame, now_frame, masked = False):
        """
        I assume here to get as input numpy arrays.
        """
        shape = prev_frame.shape
        motion_map = np.zeros_like(prev_frame, dtype='int16')

        # notice here the approach at borders, at the moment we neglect right and bottom leftovers
        for row in range(0, shape[0]-self.block_size, self.block_size):
            for col in range(0, shape[1]-self.block_size, self.block_size):
                # iterating over all image positions
                shift_distance = -1
                
                this_block = prev_frame[row:row + self.block_size, col:col+self.block_size]
                match_position = (row, col)
                path = [match_position]

                # large diamond search
                stopping_condition = False
                while(not stopping_condition):
                    shift_distance += 1
                    min_diff = float('inf')
                    best_pos = match_position
                    for offset in self.large_search_pattern_offsets:
                        (row2, col2) = (
                            match_position[0]+offset[0], match_position[1]+offset[1])
                        row2 = min(max(row2, 0), shape[0]-self.block_size-1)
                        col2 = min(max(col2, 0), shape[1]-self.block_size-1)
                        block = now_frame[row2:row2 +
                                          self.block_size, col2:col2+self.block_size]
                        diff = self.block_difference(this_block, block)
                        if(diff < min_diff):
                            min_diff = diff
                            best_pos = (row2, col2)
                    stopping_condition = (match_position == best_pos)
                    path.append(best_pos)
                    match_position = (best_pos)

                # small diamond search
                min_diff = float('inf')
                for offset in self.small_search_pattern_offsets:
                    (row2, col2) = (
                        match_position[0]+offset[1], match_position[1]+offset[0])
                    row2 = min(max(row2, 0), shape[0]-self.block_size-1)
                    col2 = min(max(col2, 0), shape[1]-self.block_size-1)
                    block = now_frame[row2:row2 +
                                      self.block_size, col2:col2+self.block_size]
                    diff = self.block_difference(this_block, block)
                    if(diff < min_diff):
                        min_diff = diff
                        best_pos = (row2, col2)
                if(best_pos != match_position):
                    shift_distance += 1
                    path.append(best_pos)

                # Add motion to all the positions matched, motion is proportional to the amount of shift
                if shift_distance:
                    for position in path:
                        motion_map[position[0]][position[1]] += shift_distance

        # Blurring
        gauss_rounds = 3
        for _ in range(gauss_rounds):
            motion_map = cv2.GaussianBlur(
                motion_map, (self.block_size, self.block_size), ((self.block_size-1)/6))
        
        # Normalization
        map_max = np.max(motion_map)
        normalized = motion_map.astype('float32')/map_max

        # Thresholding
        threshold = 1/3
        def thresher(x): return (x*(255/map_max)) if x > threshold else 0
        vthresher = np.vectorize(thresher)
        normalized = vthresher(normalized)
        if(masked):
            if self.previous_motion is None:
                self.previous_motion = (normalized.astype(bool)).astype('uint8')
            else:
                temp = normalized.astype(bool)
                normalized *= self.previous_motion
                self.previous_motion = temp

        return normalized.astype('float32')