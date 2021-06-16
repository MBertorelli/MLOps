import albumentations as A

import glob
import cv2

DATASET_NAME = "dataset_da_one"

def get_file_and_bbox():
    bounding_box = {}

    for f in glob.glob(DATASET_NAME + "/train/*/*.txt"):
        with open(f, "r") as bboxfile:
            bboxes = bboxfile.readlines()
            for idx, bbox in enumerate(bboxes):
                label = bbox.split()[0]
                bboxes[idx] = bbox.split()[1:]
                bboxes[idx].append(label)
            bboxes = [list(map(float, x)) for x in bboxes] # Convertimos a float las coord de bbox
        bounding_box[f.split('/')[-1][:-4]] = bboxes
    
    return bounding_box


transform = A.Compose([
    #A.RandomCrop(width=225, height=225),
    #A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=1),
    A.Blur(p=1),
], bbox_params=A.BboxParams(format='yolo'))

bounding_box = get_file_and_bbox()

#print(bounding_box["bike1_110_png.rf.863773b771552bc58b2b2c01a67778ef"])
#exit()
# READ BBOXES
# for item in bounding_box.items():
#     print(item)
# exit()

for img in glob.glob(DATASET_NAME + "/train/*/*.jpg"):
    print("IMAGE: ", img)
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes = bounding_box[img.split('/')[-1][:-4]]
    print("BBOXES: ", bboxes)

    sanity_check = lambda y: 1.0 if (y > 1) else (0.0 if (y < 0) else y)
    bboxes = [list(map(sanity_check, x)) for x in bboxes]
        
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    new_name = img[:-4] + "_AUGMENTED"
    original_bbox_format = list(map(lambda x: [int(x[-1:][0])] + x[:-1], bboxes))

    to_write = ""
    for i in original_bbox_format:
        to_write += str(i).replace('[', '').replace(']', '').replace(',', '') + "\n"

    with open(DATASET_NAME + "/train/labels/" + img.split('/')[-1][:-4] + "_AUGMENTED" + ".txt", "w+") as bboxfile:
        bboxfile.write(to_write)
        
    cv2.imwrite(new_name + ".jpg", transformed_image)
