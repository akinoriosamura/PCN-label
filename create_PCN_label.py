import os
import numpy as np
import cv2
import pcn
import sys
import time


def rotate(angle, center, landmark):
    rad =  - angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    landmark_ = np.asarray([(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                             M[1, 0] * x + M[1, 1] * y + M[1, 2]) for (x, y) in landmark])
    return M, landmark_


class ImageDate():
    def __init__(self, line, pcn_input_scale, output_bb_scale, imgDir, num_labels, dataset):
        line = line.strip().split()
        """
        num_labels = 98
        #0-195: landmark 坐标点  196-199: bbox 坐标点;
        #200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        #201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        #206: 图片名称
        num_labels = 68
        #0-135: landmark 坐标点  136-139: bbox 坐标点(x, y, w, h);
        #140: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        #141: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #142: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #143: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #144: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #145: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        #146: image path
        """
        if num_labels == 68:
            if dataset == "WFLW":
                line = self.remove_unuse_land(line)
            if len(line) != 147:
                import pdb;pdb.set_trace()
            assert(len(line) == 147)
            self.list = line
            self.landmark = np.asarray(list(map(float, line[:136])), dtype=np.float32).reshape(-1, 2)
            self.box = np.asarray(list(map(int, line[136:140])), dtype=np.int32)
            self.new_landmark = None
            self.new_box = None
            flag = list(map(int, line[140:146]))
            flag = list(map(bool, flag))
            self.pose = flag[0]
            self.expression = flag[1]
            self.illumination = flag[2]
            self.make_up = flag[3]
            self.occlusion = flag[4]
            self.blur = flag[5]
            self.path = os.path.join(imgDir, line[146])
            self.img = None
            self.new_img = None
            self.num_labels = num_labels
            self.is_detect = None
        elif num_labels == 98:
            assert(len(line) == 207)
            self.tracked_points = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
            self.list = line
            self.landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
            self.box = np.asarray(list(map(int, line[196:200])), dtype=np.int32)
            flag = list(map(int, line[200:206]))
            flag = list(map(bool, flag))
            self.pose = flag[0]
            self.expression = flag[1]
            self.illumination = flag[2]
            self.make_up = flag[3]
            self.occlusion = flag[4]
            self.blur = flag[5]
            self.path = os.path.join(imgDir, line[206])
            self.img = None
            self.new_img = None
            self.num_labels = num_labels
            self.is_detect = None
        else:
            print("len landmark is not invalid")
            exit()
        self.pcn_input_scale = pcn_input_scale
        self.output_bb_scale = output_bb_scale
        self.label = []

    def convert_data(self):
        # get center of landmark
        xy = np.min(self.landmark, axis=0).astype(np.int32)
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1
        center = (xy + wh / 2).astype(np.int32)

        self.img = cv2.imread(self.path)

        # くりぬきboxのサイズ
        # このboxを元にpcnをかけ回転させる
        boxsize = int(np.max(wh) * self.pcn_input_scale)
        # 顔枠の左上座標
        xy = center - boxsize // 2
        x1, y1 = xy
        # 顔枠右下の座標
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

        # 顔枠でクロップ
        imgT = self.img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            # 画像をコピーし周りに境界を作成
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_REPLICATE)

        # クロップサイズに輪郭点ラベルを合わせる
        landmark = self.landmark - xy

        # PCNで回転角を取得し正座画像に
        new_box_dict, angle = self.pcn_detect(imgT)
        # 顔枠が取れない場合はスキップ
        if angle != None:
            self.is_detect = True
            # pcnの顔枠 * output_bb_scale倍の大きさでcropし学習データ作成
            # 輪郭点が顔枠からはみ出るとうまく学習できない可能性があるため
            pcn_x = new_box_dict["x"]
            pcn_y = new_box_dict["y"]
            pcn_w = new_box_dict["w"]
            pcn_center_x = int(pcn_x + pcn_w // 2)
            pcn_center_y = int(pcn_y + pcn_w // 2)

            croped_img_w = int(pcn_w * self.output_bb_scale)
            scale_size = (croped_img_w - pcn_w) // 2
            # crop image
            croped_x1 = pcn_center_x - croped_img_w // 2
            croped_y1 = pcn_center_y - croped_img_w // 2
            croped_x2 = croped_x1 + croped_img_w
            croped_y2 = croped_y1 + croped_img_w
            self.new_img = imgT[croped_y1:croped_y2, croped_x1:croped_x2]
            # scale bbox
            new_bb_x = scale_size
            new_bb_y = scale_size
            self.new_box = [new_bb_x, new_bb_y, pcn_w, pcn_w]
            # scale landmarks
            self.new_landmark = landmark - np.array([croped_x1, croped_y1])

            # pcnの出力回転角で輪郭点と画像を回転
            xy = np.min(self.new_landmark, axis=0).astype(np.int32)
            zz = np.max(self.new_landmark, axis=0).astype(np.int32)
            wh = zz - xy + 1
            center = (xy + wh / 2).astype(np.int32)
            cx, cy = center
            M, self.new_landmark = rotate(angle, (cx, cy), self.new_landmark)
            self.new_img = cv2.warpAffine(self.new_img, M, (int(self.new_img.shape[1]), int(self.new_img.shape[0])), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            for land in self.new_landmark:
                for each_land in land:
                    self.label.append(str(each_land))
            for bb in self.new_box:
                self.label.append(str(bb))
            attributes = ["0"] * 6
            self.label.extend(attributes)
        else:
            self.is_detect = False


    def save(self, debugDir, saveName):
        savePath = os.path.join(debugDir, saveName)
        originPath = savePath + "origin.jpg"

        cv2.rectangle(self.new_img, (self.new_box[0], self.new_box[1]), (self.new_box[0] + self.new_box[2], self.new_box[1] + self.new_box[3]), (255, 0, 0), 1, 1)
        for x, y in self.new_landmark:
            cv2.circle(self.new_img, (int(x), int(y)), 3, (0, 0, 255))

        cv2.imwrite(savePath, self.new_img)
        cv2.rectangle(self.img, (self.box[0], self.box[1]), (self.box[0] + self.box[2], self.box[1] + self.box[3]), (255, 0, 0), 1, 1)
        for x, y in self.landmark:
            cv2.circle(self.img, (int(x), int(y)), 3, (0, 0, 255))
        cv2.imwrite(originPath, self.img)
        print("DEBUG: save in ", savePath)

    def pcn_detect(self, img):
        # def crop_face(img, face:Window, crop_size=200):
        winlist = []
        winlist = pcn.detect(img)
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
        else:
            print("pass get image for cannot pcn detecting in : ", os.path.basename(self.path))

        return pcn_box, angle

if __name__ == '__main__':
    # read label file
    if len(sys.argv) == 2:
        phase = sys.argv[1]
    else:
        print("please set arg(phase, ex: python create_PCN_labe.py train")
        exit()
    is_debug = True

    PCN_INPUT_SCALE = 4
    OUTPUT_PCN_BB_SCALE = 1.5

    DatasetDir = "./growing"
    imgDir = os.path.join(DatasetDir, "growing_20180601")
    labelPath = os.path.join(DatasetDir, "traindata8979_20180601_"+phase+".txt")
    outputDir = os.path.join(DatasetDir, "pcn_growing_images")
    newlabelPath = os.path.join(DatasetDir, "pcn_growing_annotaions_"+phase+".txt")
    debugDir = "./pcn_debug"
    os.makedirs(outputDir, exist_ok=True)
    os.makedirs(debugDir, exist_ok=True)

    print("convert {} to {}".format(labelPath, newlabelPath))
    print("rotate {} and save in {}".format(imgDir, outputDir))
    time.sleep(3)

    with open(labelPath, 'r') as label_f:
        lines = label_f.readlines()
        print("labels num; ", len(lines))
        processed_num = 0

        with open(newlabelPath, mode='w') as newlabel_f:
            for id, line in enumerate(lines):
                Img = ImageDate(line, PCN_INPUT_SCALE, OUTPUT_PCN_BB_SCALE, imgDir, 68, "growing")
                img_name = Img.path
                Img.convert_data()
                if not Img.is_detect:
                    continue
                saveName = str(id) + ".jpg"
                savePath = os.path.join(outputDir, saveName)
                cv2.imwrite(savePath, Img.new_img)
                Img.label.append(saveName)
                if len(Img.label) != 147:
                    import pdb;pdb.set_trace()
                assert len(Img.label) == 147
                str_label = " ".join(Img.label) + "\n"
                newlabel_f.write(str_label)
                processed_num += 1
                # print("save new label of ", img_name)
                if is_debug and id < 20:
                    Img.save(debugDir, saveName)
            print("new labels num; ", processed_num)
            print("cannot detect face by pcn: ", len(lines) - processed_num)

    print("finish create pcn labels dataset")
