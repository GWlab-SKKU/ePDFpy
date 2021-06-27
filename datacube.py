import file
import image_process

class DataCube:


    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_img = None
        self.img = None
        self.center = None

        ###################### Methods ###################
        self.get_center = lambda _self, intensity_range, step_size: image_process.get_center(self.img, intensity_range, step_size)
        self.get_azimuthal_average = lambda _self, raw_img, center: image_process.get_azimuthal_average(raw_img,center)
        self.

    def ready(self):
        pass
