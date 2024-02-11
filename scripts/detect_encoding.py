import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    confidence = result['confidence']
    return encoding, confidence

file_path = "./nano-imagenet-30/val_annotations.txt"
encoding, confidence = detect_encoding(file_path)
print("Detected encoding:", encoding)
print("Confidence:", confidence)


