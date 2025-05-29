import sys
import time
import cv2
import base64
import numpy as np
from sil_sdk.modules.vis import VISModule

vis = VISModule(server_uri="ws://0.0.0.0:50004")
vis.load("hum_pose")  # This is skipped internally for hum_pose
vis.run("hum_pose")

try:
    while True:
        result = vis.get_result("hum_pose")
        if result:
            print("Human Pose Result:", result["result"])
            image_b64 = result.get("image")
            if image_b64:
                img_data = base64.b64decode(image_b64)
                np_arr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv2.imshow("Human Pose", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print("[INFO] No result received yet.")
        time.sleep(0.05)
finally:
    vis.stop_module("hum_pose")
    cv2.destroyAllWindows()
    sys.exit(0)
