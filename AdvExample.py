import matplotlib.pyplot as plt
class AdvExample:
    def __init__(self, file: str):
        self.file = file
        try:
            open(self.file)
        except FileNotFoundError:
            print("File not found: {}".format(self.file))
            return
        self.base_label, self.attack_label, self.final_label, self.flip_threshold = self.file.split("/")[-1].split(".jpg")[0].split("_")
        self.flip_threshold = float(self.flip_threshold)
