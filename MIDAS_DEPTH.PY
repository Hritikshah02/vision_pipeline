import sys
import time
import cv2
import base64
import numpy as np
from sil_sdk.modules.vis import VISModule

vis = VISModule(server_uri="ws://0.0.0.0:50004")
vis.load("midas_depth")
vis.run("midas_depth")

try:
    while True:
        result = vis.get_result("midas_depth")
        if result:
            print("MiDaS Depth Result:", result["result"])
            image_b64 = result.get("image")
            if image_b64:
                img_data = base64.b64decode(image_b64)
                np_arr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv2.imshow("MiDaS Depth", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        time.sleep(0.05)
finally:
    vis.stop_module("midas_depth")
    cv2.destroyAllWindows()
    sys.exit(0)
