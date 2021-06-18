import os;

photo_path = './Data/photos'
label = ''
for name in sorted(os.listdir(photo_path)):
    if not name.startswith(".") and (name.endswith(".png") or name.endswith(".jpg")) and os.path.isfile(os.path.join(photo_path, name)):
        label = label + '\n' + name

with open('label.txt', 'w') as f:
    f.write(label)