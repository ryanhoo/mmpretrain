import cv2, sys, os

def process_images(dir_path):
  for root, dirs, files in os.walk(dir_path):
    print(dirs)
    for file in files:
      if file.endswith('.jpg'):
        fpath = os.path.join(root, file)
        print(fpath)
        image = cv2.imread(fpath)
        if image is None:
          os.remove(fpath)
        else:
          cv2.imwrite(fpath, image)

process_images('./data/birds_1369_val/')
process_images('./data/birds_1369_train/')


# find ./data/birds/中白鹭/ -iname "*.jpg" | xargs python test.py
