import os
import numpy as np
import cv2
import pcn
import sys
import time

# from FaceKit.PCN.PyPCN import build_init_detector, get_PCN_result, draw_result, get_label_dict


# detector = build_init_detector()

debug = True

class ImageDate():
    def __init__(self, line, output_bb_scale):
        line = line.strip().split()
        self.landmark = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
        self.label = line
        self.new_bb = []
        self.path = line[146]
        self.img = None
        self.is_detect = None
        self.output_bb_scale = output_bb_scale

    def convert_data(self):
        # PCNでbbと回転角を取得
        # new_box_dict: {'x': 273, 'y': 231, 'w': 321, 'h': 321}
        # angle: (-180, 180)
        self.img = cv2.imread(self.path)
        if self.img is not None:
            new_box_dict, angle = self.pcn_detect(imgT)

            # landmark[0]と一番近い左上bbをもつlabelを採用
            land_chin0x = self.landmark[0][0]
            label_dict = None
            angle = None
            pre_chin0 = 10000
            # import pdb;pdb.set_trace()
            for label, ang in zip(label_dicts, angles):
                diff_chin0 = abs(land_chin0x - new_box_dict["bbox"]["x"])
                if diff_chin0 <= pre_chin0:
                    label_dict = label
                    angle = ang
                    pre_chin0 = diff_chin0

            # 顔枠が取れない場合はスキップ
            # TODO: apply pcn rotate and landmark process
            if angle != None:
                self.is_detect = True
                try:
                    height, width, _ = self.img.shape
                except Exception as e:
                    import pdb;pdb.set_trace()
                # pcnの顔枠 * output_bb_scale倍の大きさでcropし学習データ作成
                # 輪郭点が顔枠からはみ出るとうまく学習できない可能性があるため
                pcn_w = new_box_dict["w"]
                new_w_size = pcn_w * self.output_bb_scale
                add_scale = new_w_size - pcn_w
                pcn_x = new_box_dict["x"] - (add_scale / 2)
                pcn_y = new_box_dict["y"] - (add_scale / 2)
                if debug:
                    # 表示して確認
                    img_tmp = self.img.copy()
                    cv2.rectangle(img_tmp, (new_box_dict["x"], new_box_dict["y"]), (new_box_dict["x"]+pcn_w, new_box_dict["y"]+pcn_w), (255, 0, 0), 1, 1)
                    for x, y in (self.landmark + 0.5).astype(np.int32):
                        cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                    cv2.imwrite("./sample_pcnimg.jpg", img_tmp)
                # 顔枠の左上 or 画像の左上縁
                dx = max(0, -pcn_x)
                dy = max(0, -pcn_y)
                x1 = int(max(0, pcn_x))
                y1 = int(max(0, pcn_y))
                x2 = pcn_x + new_w_size
                y2 = pcn_y + new_w_size
                # 顔枠の右下 or 画像の右下縁
                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = int(min(width, x2))
                y2 = int(min(height, y2))
                if debug:
                    # 表示して確認
                    img_tmp = self.img.copy()
                    cv2.rectangle(img_tmp, (x1, y1), (x2, y2), (255, 0, 0), 1, 1)
                    for x, y in (self.landmark + 0.5).astype(np.int32):
                        cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                    cv2.imwrite("./sample_newimg.jpg", img_tmp)
                self.label[136] = str(x1)
                self.label[137] = str(y1)
                self.label[138] = str(x2-x1)
                self.label[139] = str(y2-y1)
                if debug:
                    # 表示して確認
                    img_tmp = self.img.copy()
                    cv2.rectangle(img_tmp, (int(self.label[136]), int(self.label[137])), (int(self.label[136])+int(self.label[138]), int(self.label[137])+int(self.label[139])), (255, 0, 0), 1, 1)
                    for x, y in (self.landmark + 0.5).astype(np.int32):
                        cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                    cv2.imwrite("./sample_lastimg.jpg", img_tmp)
            else:
                self.is_detect = False
        else:
            self.is_detect = False
        print("is_detect: ", self.is_detect)


    def save(self, debugDir, saveName):
        savePath = os.path.join(debugDir, saveName)

        img_tmp = self.img.copy()
        cv2.rectangle(img_tmp, (int(self.label[136]), int(self.label[137])), (int(self.label[136])+int(self.label[138]), int(self.label[137])+int(self.label[139])), (255, 0, 0), 1, 1)
        for x, y in self.landmark:
            cv2.circle(img_tmp, (int(x), int(y)), 3, (0, 0, 255))
        cv2.imwrite(savePath, img_tmp)
        print("DEBUG: save in ", savePath)

    def pcn_detect(self, img):
        # def crop_face(img, face:Window, crop_size=200):
        winlist = []
        try:
            winlist = pcn.detect(img)
        except RuntimeError as e:
            print("get error: ", e)
        """
        if winlist == []:
            rotate = 20
            center = (img.shape[1]*0.5,img.shape[0]*0.5)
            rotMat = cv2.getRotationMatrix2D(center, rotate, 1.0)    
            img = cv2.warpAffine(img, rotMat, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
            print("try pcn")
        """
        pcn_box = {}
        angle = None
        if winlist != []:
            # print(winlist[0].__dict__)
            # conf = winlist[0].score
            x1 = winlist[0].x
            y1 = winlist[0].y
            width = winlist[0].width
            angle = winlist[0].angle
            pcn_box = {"x": x1, "y": y1, "w": width, "h": width}

        return pcn_box, angle

if __name__ == '__main__':
    OUTPUT_PCN_BB_SCALE = 1.5

    DatasetDir = "moru_hard_20200716/"
    labelPath = os.path.join(DatasetDir, "moru_hard_20200716.txt")
    newlabelPath = os.path.join(DatasetDir, "moru_hard_20200716_pcn.txt")

    debugDir = "./pcn_debug"
    os.makedirs(debugDir, exist_ok=True)

    print("convert {} to {}".format(labelPath, newlabelPath))
    time.sleep(3)

    with open(labelPath, 'r') as label_f:
        lines = label_f.readlines()
        print("labels num; ", len(lines))
        processed_num = 0

        with open(newlabelPath, mode='w') as newlabel_f:
            for id, line in enumerate(lines):
                if len(line.strip().split()) != 147:
                    print("len error :", line)
                    continue
                Img = ImageDate(line, OUTPUT_PCN_BB_SCALE)
                img_name = Img.path
                Img.convert_data()
                if not Img.is_detect:
                    continue
                if len(Img.label) != 147:
                    import pdb;pdb.set_trace()
                assert len(Img.label) == 147
                str_label = " ".join(Img.label) + "\n"
                newlabel_f.write(str_label)
                processed_num += 1
                # print("save new label of ", img_name)
                if debug and id < 20:
                    Img.save(debugDir, os.path.basename(img_name))
                print("num: ", id)
                #import pdb; pdb.set_trace()
            print("new labels num; ", processed_num)
            print("cannot detect face by pcn: ", len(lines) - processed_num)

    print("finish create pcn labels dataset")
