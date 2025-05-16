
from roboflow import Roboflow
rf = Roboflow(api_key="kFUOEySYQALbOC142Pj0")
project = rf.workspace("fastnuces-uakqb").project("fyp-shoplift")
version = project.version(1)
dataset = version.download("yolov11")
                