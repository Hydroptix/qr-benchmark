from abc import ABC, abstractmethod
from typing import List, Optional, Dict, TextIO, Tuple
from pathlib import Path
from pyzbar.pyzbar import ZBarSymbol
import pyzbar.pyzbar as pz
import numpy as np
import pyboof as pb
import cv2


class QRCodeScanner(ABC):
    def __init__(self, name: str):
        self.name = name


class QRCodeOpenScanner(QRCodeScanner, ABC):
    # For QR code libraries that take images opened by open()

    @abstractmethod
    def scan(self, img: TextIO) -> Tuple[int, int]:
        pass


class QRCodeCVImgScanner(QRCodeScanner, ABC):
    # For QR code libraries that take images opened by cv2.imread()

    @abstractmethod
    def scan(self, mat) -> Tuple[int, int]:
        pass


class QRCodePathScanner(QRCodeScanner, ABC):
    # For QR code libraries that need custom handling from an image path

    @abstractmethod
    def scan(self, img_path: str) -> Tuple[int, int]:
        pass


class QRCodeScanResults:
    def __init__(self, qr_scanner: QRCodeScanner):
        self.scanner = qr_scanner.name
        self.detected = 0
        self.decoded = 0


class OpenCVDetector(QRCodeCVImgScanner):
    def __init__(self, name: str = "OpenCV QRCodeDetector"):
        super().__init__(name)
        self.cv2decoder = cv2.QRCodeDetector()

    def scan(self, mat):
        decoder = self.cv2decoder

        any_found, detected_codes = decoder.detectMulti(mat)

        if not any_found:
            return (0, 0)

        _, decoded_vals, _ = decoder.decodeMulti(mat, detected_codes)

        return len(detected_codes), len(decoded_vals) - decoded_vals.count('')


class ZBarDetector(QRCodeCVImgScanner):
    def __init__(self, name: str = "ZBar"):
        super().__init__(name)

    def scan(self, mat):
        decoded = pz.decode(mat, symbols=[ZBarSymbol.QRCODE])

        return len(decoded), len(decoded)

# Requires Java 14 or newer as default JRE
class BoofCVDetector(QRCodePathScanner):
    def __init__(self, name: str = "BoofCV"):
        super().__init__(name)

    def scan(self, img_path):
        detector = pb.FactoryFiducial(np.uint8).qrcode()

        #Load image in grayscale
        bcv_img = pb.load_single_band(img_path, np.uint8)

        detector.detect(bcv_img)

        if(len(detector.detections) > 0):
            print(detector.detections[0].message)

        return len(detector.detections) + len(detector.failures), len(detector.detections)


def benchmark_image(image_path: str,
                    open_scanners: Optional[List[QRCodeOpenScanner]],
                    opencv_scanners: Optional[List[QRCodeCVImgScanner]],
                    path_scanners: Optional[List[QRCodePathScanner]]) \
        -> Optional[List[QRCodeScanResults]]:
    try:
        image_purepath = Path(image_path)
    except ValueError:
        print(f"{image_path} is not a path")
        return None

    if not image_purepath.exists():
        print(f"{str(image_path)} does not exist")

    # If None is passed in for one of the arguments, initialize it to the default list of scanners
    if open_scanners is None:
        open_scanners = []

    if opencv_scanners is None:
        opencv_scanners = [OpenCVDetector(), ZBarDetector()]

    if path_scanners is None:
        path_scanners = [BoofCVDetector()]

    # Initialize results dict
    scan_results: Dict[str, QRCodeScanResults] = {}

    for open_scanner in open_scanners:
        scan_results[open_scanner.name] = QRCodeScanResults(open_scanner)

    for opencv_scanner in opencv_scanners:
        scan_results[opencv_scanner.name] = QRCodeScanResults(opencv_scanner)

    for path_scanner in path_scanners:
        scan_results[path_scanner.name] = QRCodeScanResults(path_scanner)

    if len(open_scanners) > 0:
        try:
            img = open(image_path)
        except FileNotFoundError:
            print(f"{image_path} could not be opened")
            return None

        for open_scanner in open_scanners:
            detected: int = 0
            decoded: int = 0

            img.seek(0)
            detected, decoded = open_scanner.scan(img)

            scan_results[open_scanner.name].detected += detected
            scan_results[open_scanner.name].decoded += decoded

        img.close()

    if len(opencv_scanners) > 0:
        try:
            cv_img: cv2 = cv2.imread(image_path)
        except FileNotFoundError:
            print(f"{image_path} could not cv opened")
            return None

        for opencv_scanner in opencv_scanners:
            detected: int = 0
            decoded: int = 0

            detected, decoded = opencv_scanner.scan(cv_img)

            scan_results[opencv_scanner.name].detected += detected
            scan_results[opencv_scanner.name].decoded += decoded

        # Allow garbage collection to free this if necessary
        cv_img = None

    for path_scanner in path_scanners:
        detected: int = 0
        decoded: int = 0

        detected, decoded = path_scanner.scan(image_path)

        scan_results[path_scanner.name].detected += detected
        scan_results[path_scanner.name].decoded += decoded

    return list(scan_results.values())


if __name__ == '__main__':
    pb.init_memmap()

    results = benchmark_image(r"C:\Users\sfrazee\Downloads\labelbox_upload\DJI_0294\DJI_0294-0h0m0s0.png",
                              None, None, None)

    for result in results:
        print(f"{result.scanner}: {result.detected} detected, {result.decoded} decoded")
