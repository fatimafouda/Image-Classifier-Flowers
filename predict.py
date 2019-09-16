'''
Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
'''
import argparse
from pred_utils import load_checkpoint, preproc_image, predict, flower_type
from utilis import read_labels

parser = argparse.ArgumentParser(description="predict class plus optionally topk classes of an input image")
parser.add_argument("image_path", default="flowers/test/11/image_03147.jpg", help="path to image")
parser.add_argument("checkpoint", default="train_checkpoint.pth", help="path to checkpoint/classifier params")
parser.add_argument("--top_k", type=int, default=1, help="top propable classes")
parser.add_argument("--labels", help="path to json file containing labels")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")

args = parser.parse_args()

print("Loading classifier..")
model= load_checkpoint(args.checkpoint)

print("Preprocessing input image..")
image = preproc_image(args.image_path)

print("Classifying image..")
probs, labels = predict(image, model, args.top_k, args.gpu)

if args.labels:
    print("Loading categories..")
    cats = read_labels(args.labels)
    flowers =  flower_type(labels, cats)