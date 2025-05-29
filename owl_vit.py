import sys
import time
import cv2
import base64
import numpy as np
from sil_sdk.modules.vis import VISModule

module = "owl_vit"
prompt = "find all persons in the frame"

vis = VISModule(server_uri="ws://0.0.0.0:50004", start_server=False)
vis.load(module)
vis.run(module, prompt=prompt)

while True:
    result = vis.get_result(module)
    if result:
        print("Detection Result:", result.get("result"))

        image_b64 = result.get("image")
        if image_b64:
            try:
                img_data = base64.b64decode(image_b64)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("[ERROR] cv2.imdecode failed — got None")
                else:
                    cv2.imshow(f"{module} Output", frame)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
            except Exception as e:
                print(f"[ERROR] Image decoding or display failed: {e}")
        else:
            print("[INFO] No image in result.")
    time.sleep(0.05)

cv2.destroyAllWindows()
vis.stop_module(module)
vis.close()
sys.exit(0)
