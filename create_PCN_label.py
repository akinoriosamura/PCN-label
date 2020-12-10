import os
import numpy as np
import cv2
import pcn
import sys
import time

# from FaceKit.PCN.PyPCN import build_init_detector, get_PCN_result, draw_result, get_label_dict


# detector = build_init_detector()

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
    def __init__(self, line, default_pcn_input_scale, output_scale, imgDir, num_labels, dataset, pcn_type, use_prelabel):
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
            self.pcn_box = None
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
            self.pcn_img = None
            self.new_img = None
            self.num_labels = num_labels
            self.is_detect = False
            self.use_prelabel = use_prelabel
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
            self.is_detect = False
        else:
            print("len landmark is not invalid")
            exit()
        self.default_pcn_input_scale = default_pcn_input_scale
        self.output_scale = output_scale
        self.pcn_type = pcn_type
        self.label = []

    def convert_data(self):
        start = time.time()
        # get center of landmark
        xy = np.min(self.landmark, axis=0).astype(np.int32)
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1
        center = (xy + wh / 2).astype(np.int32)

        print("process img: ", os.path.basename(self.path))
        self.img = cv2.imread(self.path)
        print("in shape: ", self.img.shape)
        debug = True
        if debug:
            # 表示して確認
            img_tmp = self.img.copy()
            for x, y in (self.landmark + 0.5).astype(np.int32):
                cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
            cv2.imwrite("./sample_resized.jpg", img_tmp)
        # pcnで処理するbbに対する画像サイズ
        # 1.5(default), 2, 3, 4, 5で当たれば終了
        pcn_in_scales = [1.5, 1.2, 2]
        for in_s in pcn_in_scales:
            if self.is_detect:
                break
            boxsize = int(np.max(wh) * in_s)
            # 顔枠の左上座標
            xy = center - boxsize // 2
            x1, y1 = xy
            diff_x1, diff_y1 = x1, y1
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
            if debug:
                # 表示して確認
                imgT_tmp = imgT.copy()
                cv2.imwrite("./sample_croped.jpg", imgT_tmp)
            # クロップサイズに輪郭点ラベルを合わせる
            landmark = self.landmark - xy

            # PCNで回転角を取得し正座画像に
            # TODO: change pcn model
            new_box_dict, angle = self.pcn_detect(imgT)
            # 顔枠が取れない場合はスキップ
            # TODO: apply pcn rotate and landmark process
            if angle != None:
                self.is_detect = True
                print("in scale " + str(in_s) + " size get pcn bb")
                # pcnの顔枠 * output_scale倍の大きさでcropし学習データ作成
                # 輪郭点が顔枠からはみ出るとうまく学習できない可能性があるため
                pcn_x = new_box_dict["x"]
                pcn_y = new_box_dict["y"]
                pcn_w = new_box_dict["w"]
                pcn_center_x = int(pcn_x + pcn_w // 2)
                pcn_center_y = int(pcn_y + pcn_w // 2)
                # original画像から見ての座標
                original_pcn_x = diff_x1 + pcn_x
                original_pcn_y = diff_y1 + pcn_y
                self.pcn_img = imgT.copy()
                self.pcn_box = [pcn_x, pcn_y, pcn_x+pcn_w, pcn_y+pcn_w]
                croped_img_w = int(pcn_w * self.output_scale)
                print("croped_img_w: ", croped_img_w)
                scale_size = (croped_img_w - pcn_w) // 2
                # crop image
                croped_x1 = original_pcn_x - scale_size
                croped_y1 = original_pcn_y - scale_size
                diff_crop_x = croped_x1
                diff_crop_y = croped_y1
                croped_x2 = croped_x1 + croped_img_w
                croped_y2 = croped_y1 + croped_img_w
                img_height, img_width, _ = self.img.shape
                # 顔枠の左上 or 画像の左上縁
                croped_dx = max(0, -croped_x1)
                croped_dy = max(0, -croped_y1)
                croped_x1 = max(0, croped_x1)
                croped_y1 = max(0, croped_y1)
                croped_x2 += croped_dx
                croped_y2 += croped_dy
                # 顔枠の右下 or 画像の右下縁
                croped_edx = max(0, croped_x2 - img_width)
                croped_edy = max(0, croped_y2 - img_height)
                target_img = self.img.copy()
                if (croped_dx > 0 or croped_dy > 0 or croped_edx > 0 or croped_edy > 0):
                    # 足りない分は画像をコピーし周りに境界を作成
                    target_img = cv2.copyMakeBorder(target_img, croped_dy, croped_edy, croped_dx, croped_edx, cv2.BORDER_REPLICATE)

                self.new_img = target_img[croped_y1:croped_y2, croped_x1:croped_x2]
                if debug:
                    # 表示して確認
                    img_tmp = self.new_img.copy()
                    cv2.imwrite("./sample_new_croped.jpg", img_tmp)
                # scale bbox
                new_bb_x = original_pcn_x - diff_crop_x
                new_bb_y = original_pcn_y - diff_crop_y
                new_bb_x2 = new_bb_x + pcn_w
                new_bb_y2 = new_bb_y + pcn_w
                self.new_box = [new_bb_x, new_bb_y, new_bb_x2, new_bb_y2]
                # scale landmarks
                self.new_landmark = self.landmark - np.array([diff_crop_x, diff_crop_y])
                if debug:
                    # 表示して確認
                    try:
                        img_tmp = self.new_img.copy()
                        cv2.rectangle(img_tmp, (self.new_box[0], self.new_box[1]), (self.new_box[2], self.new_box[3]), (255, 0, 0), 1, 1)
                        for x, y in (self.new_landmark + 0.5).astype(np.int32):
                            cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                        cv2.imwrite("./sample_preprocessed.jpg", img_tmp)
                    except:
                        import pdb;pdb.set_trace()
                if not (self.new_landmark >= 0).all():
                    print("have minus value")
                    # 表示して確認
                    img_tmp = self.new_img.copy()
                    for x, y in (self.new_landmark + 0.5).astype(np.int32):
                        cv2.circle(img_tmp, (x, y), 2, (255, 0, 0))
                    cv2.imwrite("./sample_resized_9.jpg", img_tmp)
                    self.is_detect = False
                else:
                    if self.pcn_type == "rotate":
                        # pcnの出力回転角で輪郭点と画像を回転
                        xy = np.min(self.new_landmark, axis=0).astype(np.int32)
                        zz = np.max(self.new_landmark, axis=0).astype(np.int32)
                        wh = zz - xy + 1
                        center = (xy + wh / 2).astype(np.int32)
                        cx, cy = center
                        if new_bb_x < cx < new_bb_x2:
                            M, self.new_landmark = rotate(angle, (cx, cy), self.new_landmark)
                            try:
                                self.new_img = cv2.warpAffine(self.new_img, M, (int(self.new_img.shape[1]), int(self.new_img.shape[0])), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                                # get label
                                for land in self.new_landmark:
                                    for each_land in land:
                                        self.label.append(str(each_land))
                                for bb in self.new_box:
                                    self.label.append(str(bb))
                                attributes = ["0"] * 6
                                self.label.extend(attributes)
                            except:
                                self.is_detect = False
                        else:
                            self.is_detect = False
                    elif self.pcn_type == "landmark":
                        try:
                            for land in self.new_landmark:
                                for each_land in land:
                                    self.label.append(str(each_land))
                            for bb in self.new_box:
                                self.label.append(str(bb))
                            attributes = ["0"] * 6
                            self.label.extend(attributes)
                            # add pcn landmerk
                            pcn_landmark = [
                                label_dict["landmark"]["nose"][0],
                                label_dict["landmark"]["nose"][1],
                                label_dict["landmark"]["eye_left"][0],
                                label_dict["landmark"]["eye_left"][1],
                                label_dict["landmark"]["eye_right"][0],
                                label_dict["landmark"]["eye_right"][1],
                                label_dict["landmark"]["mouth_left"][0],
                                label_dict["landmark"]["mouth_left"][1],
                                label_dict["landmark"]["mouth_right"][0],
                                label_dict["landmark"]["mouth_right"][1]
                            ]
                            for i in range(9):
                                label = "chin_" + str(i)
                                pcn_landmark.extend(label_dict["landmark"][label])
                            pcn_landmark = np.asarray(pcn_landmark, dtype=np.float32).reshape(-1, 2)
                            pcn_landmark = pcn_landmark - np.array([croped_x1, croped_y1])
                            pcn_landmark = pcn_landmark.reshape(-1)
                            pcn_landmark = [str(int(land)) for land in pcn_landmark]
                            self.label.extend(pcn_landmark)
                            # print("label: ", self.label)
                        except:
                            self.is_detect = False
                    else:
                        exit()
            else:
                print("in scale " + str(in_s) + " size cant get pcn bb, next pcn input scale")
                self.is_detect = False
        if not self.is_detect:
            print("pass get image for cannot pcn detecting in : ", os.path.basename(self.path))
        print("elapsed time: ", time.time() - start)
        print("is_detect: ", self.is_detect)
        if self.use_prelabel and not self.is_detect:
            import pdb; pdb.set_trace()
            # get label
            self.new_img = self.img
            self.new_landmark = self.landmark
            self.new_box = self.box
            for land in self.new_landmark:
                for each_land in land:
                    self.label.append(str(each_land))
            for bb in self.new_box:
                self.label.append(str(bb))
            attributes = ["0"] * 6
            self.label.extend(attributes)
            self.is_detect = True


    def remove_unuse_land(self, line):
        del_ago = [2, 4, 6, 8, 9, 11, 12, 14, 18, 20, 21, 23, 24, 26, 28, 30]
        del_left_eye_blow = [38, 39, 40, 41]
        del_right_eye_blow = [47, 48, 49, 50]
        del_eye = [62, 66, 70, 74]
        del_eye_center = [96, 97]
        dels = del_ago + del_left_eye_blow + del_right_eye_blow + del_eye + del_eye_center
        # 削除する際にインデックスがずれないように降順に削除していく
        dels.sort(reverse=True)
        for del_id in dels:
            del_id_y = del_id * 2 + 1
            del_id_x = del_id * 2
            line.pop(del_id_y)
            line.pop(del_id_x)

        return line

    def save(self, debugDir, saveName):
        savePath = os.path.join(debugDir, saveName)
        pcnPath = savePath + "pcn.jpg"
        originPath = savePath + "origin.jpg"

        # save new img and label
        cv2.rectangle(self.new_img, (self.new_box[0], self.new_box[1]), (self.new_box[2], self.new_box[3]), (255, 0, 0), 1, 1)
        for x, y in self.new_landmark:
            cv2.circle(self.new_img, (int(x), int(y)), 3, (0, 0, 255))
        if not (self.new_landmark >= 0).all():
            print("have minus value")
            import pdb;pdb.set_trace()
        cv2.imwrite(savePath, self.new_img)
        # save pcn img and bb
        cv2.rectangle(self.pcn_img, (self.pcn_box[0], self.pcn_box[1]), (self.pcn_box[2], self.pcn_box[3]), (255, 0, 0), 1, 1)
        cv2.imwrite(pcnPath, self.pcn_img)
        # save original img and label
        cv2.rectangle(self.img, (self.box[0], self.box[1]), (self.box[2], self.box[3]), (255, 0, 0), 1, 1)
        for x, y in self.landmark:
            cv2.circle(self.img, (int(x), int(y)), 3, (0, 0, 255))
        cv2.imwrite(originPath, self.img)
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
    # read label file
    if len(sys.argv) == 3:
        # pcn_type:
        # rotate = rotate face and save pcn label. 
        # landmark = get pcn landmark and save no rotate.
        pcn_type = sys.argv[1]
        # (bool 0 or 1): use pre label if pcn cannnot detect
        use_prelabel = bool(int(sys.argv[2]))
    else:
        print("please set arg(phase, ex: python create_PCN_labe.py landmark")
        exit()
    is_debug = True
    print("pcn_type: ", pcn_type)
    print("use_prelabel: ", use_prelabel)

    DEFAULT_PCN_INPUT_SCALE = 1.5
    # output img scale size based on pcn bb
    OUTPUT_SCALE = 2.0
    # in DEFAULT_PCN_INPUT_SCALE = 3 then, train data is below
    # new labels num;  5538
    # cannot detect face by pcn:  2279
    # test
    # new labels num;  613
    # cannot detect face by pcn:  256
    
    """
    dataset = "growing"
    DatasetDir = "./growing"
    imgDir = os.path.join(DatasetDir, "growing_20180601")
    labelPath = os.path.join(DatasetDir, "traindata8979_20180601_"+phase+".txt")
    if pcn_type == "rotate":
        outputDir = os.path.join(DatasetDir, "rotate_pcn_growing_images")
        newlabelPath = os.path.join(DatasetDir, "rotate_pcn_growing_annotaions_"+phase+".txt")
    elif pcn_type == "landmark":
        outputDir = os.path.join(DatasetDir, "landmark_pcn_growing_images")
        newlabelPath = os.path.join(DatasetDir, "landmark_pcn_growing_annotaions_"+phase+".txt") 
    else:
        break

    dataset = "WFLW"
    DatasetDir = "./WFLW"
    imgDir = os.path.join(DatasetDir, "WFLW_images")
    phases = ["test", "train"]
    for phase in phases:
        print("phase: ", phase)
        labelPath = os.path.join(DatasetDir, "WFLW_annotations", "list_98pt_rect_attr_train_test/list_98pt_rect_attr_"+phase+".txt")
        if pcn_type == "rotate":
            outputDir = os.path.join(DatasetDir, "rotate_pcn_WFLW68_images")
            newlabelPath = os.path.join(DatasetDir, "WFLW_annotations", "rotate_pcn_WFLW68_annotaions_"+phase+".txt")
        elif pcn_type == "landmark":
            outputDir = os.path.join(DatasetDir, "add_pcn_landmark_WFLW68_images")
            newlabelPath = os.path.join(DatasetDir, "WFLW_annotations", "add_pcn_landmark_WFLW68_annotaions_"+phase+".txt") 
        else:
            print("error")
            exit()

    dataset = "baobab"
    DatasetDir = "/Users/osamura/Documents/jolijoli/dataset/alignment/BAOBAB/baobab_0604"
    imgDir = os.path.join(DatasetDir, "img")
    labelPath = os.path.join(DatasetDir, "json/baobab_0604.txt")
    outputDir = os.path.join(DatasetDir, "pcn_img")
    newlabelPath = os.path.join(DatasetDir, "json/baobab_0604_pcn.txt")

    dataset = "baobab"
    DatasetDir = "./moru_hard_20200716"
    imgDir = os.path.join(DatasetDir, "img")
    labelPath = os.path.join(DatasetDir, "moru_hard_20200716.txt")
    outputDir = os.path.join(DatasetDir, "pcn_img")
    newlabelPath = os.path.join(DatasetDir, "moru_hard_20200716_pcn.txt")

    dataset = "growing"
    DatasetDir = "./growing"
    imgDir = os.path.join(DatasetDir, "growing_20180601")
    labelPath = os.path.join(DatasetDir, "traindata8979_20180601.txt")
    outputDir = os.path.join(DatasetDir, "pcn_growing_20180601")
    newlabelPath = os.path.join(DatasetDir, "traindata8979_20180601_pcn.txt")
    """

    dataset = "alldata"
    DatasetDir = "/data/landmarks_datasets/merged/300W_COFW_Menpo2D_MultiPIE_xm2vts_wflw_growing/"
    imgDir = "/data/landmarks_datasets/"
    labelPath = os.path.join(DatasetDir, "all_68_rect_attr.txt")
    outputDir = os.path.join(DatasetDir, "pcn_imgs")
    newlabelPath = os.path.join(DatasetDir, "pcn_all_68_rect_attr.txt")

    debugDir = "./pcn_debug"
    os.makedirs(outputDir, exist_ok=True)
    os.makedirs(debugDir, exist_ok=True)

    print("convert {} to {}".format(labelPath, newlabelPath))
    # print("rotate {} and save in {}".format(imgDir, outputDir))
    time.sleep(3)

    with open(labelPath, 'r') as label_f:
        lines = label_f.readlines()
        print("labels num; ", len(lines))
        processed_num = 0

        with open(newlabelPath, mode='w') as newlabel_f:
            for id, line in enumerate(lines):
                Img = ImageDate(line, DEFAULT_PCN_INPUT_SCALE, OUTPUT_SCALE, imgDir, 68, dataset, pcn_type, use_prelabel)
                img_name = Img.path
                # if "115" not in img_name:
                #     continue
                Img.convert_data()
                # import pdb; pdb.set_trace()
                if not Img.is_detect:
                    continue
                saveName = "pcn_" + os.path.basename(img_name)
                savePath = os.path.join(outputDir, saveName)
                cv2.imwrite(savePath, Img.new_img)
                Img.label.append(saveName)
                if pcn_type == "rotate":
                    if len(Img.label) != 147:
                        import pdb;pdb.set_trace()
                    assert len(Img.label) == 147
                elif pcn_type == "landmark":
                    if len(Img.label) != 175:
                        import pdb;pdb.set_trace()
                    assert len(Img.label) == 175
                str_label = " ".join(Img.label) + "\n"
                newlabel_f.write(str_label)
                processed_num += 1
                # print("save new label of ", img_name)
                # if "115" in saveName:
                #     import pdb; pdb.set_trace()
                #     Img.save(debugDir, saveName)
                if is_debug and id < 50:
                    # print("label: ", str_label)
                    Img.save(debugDir, saveName)
            print("new labels num; ", processed_num)
            print("cannot detect face by pcn: ", len(lines) - processed_num)

    print("finish create pcn labels dataset")
