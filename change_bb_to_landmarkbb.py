# coding:utf-8

import os
import numpy as np
import cv2
import sys
import time

debug = True


class ImageDate():
    def __init__(self, line, output_bb_scale):
        line = line.strip().split()
        self.landmark = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
        self.label = line
        self.new_bb = []
        self.path = line[146]
        self.img = None
        self.is_detect = True
        self.output_bb_scale = output_bb_scale

    def convert_data(self):
        # PCNでbbと回転角を取得
        # new_box_dict: {'x': 273, 'y': 231, 'w': 321, 'h': 321}
        # angle: (-180, 180)
        self.img = cv2.imread(self.path)
        if self.img is not None:
            xy = np.min(self.landmark, axis=0).astype(np.int32)
            zz = np.max(self.landmark, axis=0).astype(np.int32)
            wh = zz - xy + 1
            center = (xy + wh / 2).astype(np.int32)

            # 顔枠のサイズ
            boxsize = int(np.max(wh) * self.output_bb_scale)
            # 顔枠の左上を原点とした中心までの座標
            xy = center - boxsize // 2
            x1, y1 = xy
            x2, y2 = xy + boxsize
            try:
                height, width, _ = self.img.shape
            except Exception as e:
                import pdb;pdb.set_trace()
            # 顔枠の左上 or 画像の左上縁
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            # 顔枠の右下 or 画像の右下縁
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            self.label[136] = str(x1)
            self.label[137] = str(y1)
            self.label[138] = str(x2)
            self.label[139] = str(y2)
            if debug:
                # 表示して確認
                img_tmp = self.img.copy()
                cv2.rectangle(img_tmp, (int(self.label[136]), int(self.label[137])), (int(self.label[136])+int(self.label[138]), int(self.label[137])+int(self.label[139])), (255, 0, 0), 1, 1)
                for x, y in (self.landmark + 0.5).astype(np.int32):
                    cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                cv2.imwrite("./sample_lastimg.jpg", img_tmp)
        else:
            self.is_detect = False
        # print("is_detect: ", self.is_detect)


    def save(self, debugDir, saveName):
        savePath = os.path.join(debugDir, saveName)

        img_tmp = self.img.copy()
        cv2.rectangle(img_tmp, (int(self.label[136]), int(self.label[137])), (int(self.label[136])+int(self.label[138]), int(self.label[137])+int(self.label[139])), (255, 0, 0), 1, 1)
        for x, y in self.landmark:
            cv2.circle(img_tmp, (int(x), int(y)), 3, (0, 0, 255))
        cv2.imwrite(savePath, img_tmp)
        print("DEBUG: save in ", savePath)


if __name__ == '__main__':
    OUTPUT_BB_SCALE = 1.3

    DatasetDir = "/data/clev2_grow_helen_wflw_68/"
    labelPath = os.path.join(DatasetDir, "nonpcn_aug_clev2_grow_helen_wflw_68.txt")
    newlabelPath = os.path.join(DatasetDir, "landbb_aug_clev2_grow_helen_wflw_68.txt")

    debugDir = "./landmarkbb_debug"
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
                Img = ImageDate(line, OUTPUT_BB_SCALE)
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
                if id % 1000 == 0:
                    print("num: ", id)
                #import pdb; pdb.set_trace()
            print("new labels num; ", processed_num)
            print("cannot detect face by pcn: ", len(lines) - processed_num)

    print("finish create pcn labels dataset")
