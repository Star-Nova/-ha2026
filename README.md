!pip install roboflow from roboflow import Roboflow rf = Roboflow(api_key="yULE7Bm4YBBtCpJjpYVf") project = rf.workspace("kodyazari").project("weagle-iha-nesne-tespiti-omdad") version = project.version(2) dataset = version.download("yolov8")


https://universe.roboflow.com/kodyazari/weagle-iha-nesne-tespiti-omdad
