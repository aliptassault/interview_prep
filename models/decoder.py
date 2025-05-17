import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        detector = chardet.universaldetector.UniversalDetector()
        for line in file:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']

if __name__ == "__main__":
    file_path = "./enterprise_docs/test1.pdf"
    encoding = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")