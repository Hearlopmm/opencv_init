# coding=UTF-8
"""用于图像的初始化"""
import os
import cv2
import numpy
import numpy as np
import time
import Serial as se


# 定义一个函数，显示图片
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 平均整张图片光照，解决光照不均衡产生巨大噪点
def unevenLightCompensate(img, blockSize=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if rowmax > gray.shape[0]:
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if colmax > gray.shape[1]:
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst


# 灰度图、二值化、高斯滤波预处理 结果黑底白色
def predeal(frame, threshold_thresh=100, kernel_shape=3, blur=True, thr_show=False, filter_show=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval1, dst1 = cv2.threshold(gray, threshold_thresh, 255, cv2.THRESH_BINARY)
    # 二值化：高于阈值的变成白色（设置的255），其他变成黑色
    if blur:
        dst1 = cv2.medianBlur(dst1, 7)  # 平滑
    kernel = np.ones((kernel_shape, kernel_shape), np.uint8)
    mask = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, kernel)  # opening:消除细小物体
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # closing
    # mask = cv2.dilate(mask, (8, 8))
    if thr_show:
        cv2.imshow("threshold", dst1)
        cv2.waitKey(0)
    if filter_show:
        cv2.imshow('smooth_maybe the error is on filter not threshold!', mask)
        cv2.waitKey(0)
    return mask


# !unchangeable
# 检测角点--k敏反比
# inimg:BGR_image|img:gray_image
# inimg must have the same size with img
def CornerHarris(inimg, img, threshold=0.01, show=False):
    img = np.float32(img)
    dst = cv2.cornerHarris(img, 2, 3, 0.08)  # 搞出了一个数组，角点响应值高
    dst = cv2.dilate(dst, None)
    inimg[dst > threshold * dst.max()] = [0, 0, 0]
    if show:
        cv_show('afterHarris', inimg)
    return inimg


# 找中点
def find_center(img):
    h, w = img.shape[:2]
    y_center = h / 2
    x_center = w / 2
    return y_center, x_center


# !unchangeable
# 缩放--新宽度
def imgResize(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (new_w, new_h))
    return resized_img


# 锐化
# power1越小锐化越厉害，且相当灵敏，调上下0.05
# 锐化程度高时，请使二值化保留更多细节
def gama_transfer(img, power1):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = 255 * np.power(img / 255, power1)
    img = np.around(img)
    img[img > 255] = 255
    out_img = img.astype(np.uint8)
    return out_img


# 去除较大的白色区域
def move_whitePattern(mask, moveSize_proportion=10, show=False):  # 输入黑底白色物体图
    imgcanny = cv2.Canny(mask, 50, 100)
    img_size = min(mask.shape[0], mask.shape[1])
    fcon, hier = cv2.findContours(imgcanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in fcon:
        x, y, w, h = cv2.boundingRect(c)
        if w < img_size / moveSize_proportion or h < img_size / moveSize_proportion:
            mask[y:y + h, x:x + w] = 0
            # continue
        # cv2.rectangle(mask, (x, y), (x + w, y + h), 255, 1)
    if show:
        cv_show('moveWhiteDot', mask)
    return mask


# fps
# 初始化要让两个fps_pTime = fps_cTime = 0
def get_fps(show=False, inimg=None):
    global fps_pTime, fps_cTime
    fps_cTime = time.time()
    fps = 1 / (fps_cTime - fps_pTime)
    fps_pTime = fps_cTime
    if show:
        cv2.putText(inimg, f"fps: {int(fps)}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)
    return fps


# 自己的滤波器 —————————————————————————————————————————————————————————————————————
# use:实例化对象——创建目标矩阵——apply
# Filter = init.Filter()
# dst_img = numpy.zeros_like(img)
# Filter.apply(img, dst_img)
class VConvFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


class Filter(VConvFilter):
    def __init__(self):
        # 奇数行列、所有值求和为1
        kernel = numpy.array([[-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, 49, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1]])
        VConvFilter.__init__(self, kernel)


# ————————————————————————————————————————————————————————————————————————

# 把虚线变成实线_方法1：用了角点检测 robust but slow
# img:haved predealed
def dealDottedLine1(img, show=False):
    h, w = img.shape[:2]
    img = imgResize(img, 200)
    bgrimage = CornerHarris(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), img, threshold=0.01)
    bgrimage = ~bgrimage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bgrimage = cv2.dilate(bgrimage, kernel)
    # cv_show('wtf', bgrimage)
    bgrimage = cv2.medianBlur(bgrimage, 7)  # 平滑：耗时长就注释
    bgrimage = ~bgrimage
    out = cv2.resize(bgrimage, (w, h))
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = cv2.medianBlur(out, 7)  # 平滑：耗时长就注释
    if show:
        cv_show('deal dotted line', out)
    return out


# 把虚线变成实线_方法2：可能速度快但鲁棒性低（需要调参） ！优先选择
# img:haved predealed
def dealDottedLine2(img, show=False):
    h, w = img.shape[:2]
    img = imgResize(img, 200)
    dst1 = ~img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.dilate(dst1, kernel)
    image = cv2.medianBlur(image, 7)  # 平滑：耗时长就注释
    out = cv2.resize(image, (w, h))
    out = cv2.medianBlur(out, 7)  # 平滑：耗时长就注释
    retval1, out = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY_INV)
    if show:
        cv_show('deal dotted line', out)
    return out


# hsv筛选颜色--这个函数也没啥必要
# img:hsv图片，不需要预处理
def hsvmask(img, upper=None, lower=None, show=False):
    if lower is None:
        lower = np.array([255, 255, 255])
    else:
        lower = np.array(lower)
    if upper is None:
        upper = np.array([255, 255, 255])
    else:
        upper = np.array(upper)
    mask = cv2.inRange(img, lower, upper)
    if show:
        cv_show('hsv_mask', mask)
    return mask


# 滤出红色
# img_block1 = cv2.inRange(img_hsvBlock, red1_min, red1_max)
# img_block2 = cv2.inRange(img_hsvBlock, red2_min, red2_max)
# img_blockB = cv2.bitwise_or(img_block1, img_block2)


# 画轮廓，白色物体黑色背景，如果不是canny选True，all选False，具体见笔记。
# draw=True,inimg:BGR图像
def Contours(img, whet_canny=False, allTheContours=True, draw=False, inimg=None):
    if whet_canny:
        img = cv2.Canny(img, 50, 100)
    if allTheContours:
        fcon, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    else:
        fcon, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("num_contours=", len(fcon))
    if draw:
        cv2.drawContours(inimg, fcon, -1, (0, 255, 0), 2)
        cv_show("drawContours", inimg)
    return fcon


def contour2shape(contour):
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    # 使用多边形逼近轮廓的形状
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)  # 这个参数越小点越多
    # 获取角点坐标, epsilon是拟合线的最大距离
    cornerNum = len(approx)  # 角点数
    return cornerNum


# -----------------模板匹配----------------------
def init_modelFile(modelIndex, modelImages, path=None):
    myList = os.listdir(path)
    for n in myList:
        img = cv2.imread(f'{path}/{n}', 0)
        modelImages.append(img)
        modelIndex.append(os.path.splitext(n)[0])
    return modelIndex, modelImages


# img:gray图
# resize调整图片大小与模板相同（ORB尺度不变性不好）
def ORB_match(img, modelIndex, modelImages, modelH=150, ORBthread=0.9):
    img = imgResize(img, modelH)
    # cv_show('model', img)
    source = []
    for n in modelImages:
        res = cv2.matchTemplate(img, n, cv2.TM_CCOEFF_NORMED)
        max_val = cv2.minMaxLoc(res)[1]
        source.append(max_val)
    digit = source.index(max(source))
    name = modelIndex[digit]
    # print(source)
    if max(source) > ORBthread:
        return ''.join(name)
    else:
        return None


# using：
# index = []
# models = []
# index, models= init.init_modelFile(index, models, path='Number')
# img = cv2.imread("Number/2.jpg")
# img = init.predeal(img)
# p = init.ORB_match(img, index, models, resize=False)
# print(p)
# ------------------------------------------------------


class MessageFormat(object):  # 通讯格式打包
    def __init__(self):
        self.start = '@'
        self.stD = '@D'  # 发送数据
        self.stC = '@C'  # 发送指令
        self.end = '#'

    def MesSendD(self, mes_single, message):
        return f"{self.stD}{mes_single}:{message}{self.end}"

    def MesSendC(self, mes_single, message):
        return f"{self.stC}{mes_single}:{message}{self.end}"


# 就...存下来看看大小
def save2ss_size(img):
    cv2.imwrite('t_size' + str(int(time.time())) + '.jpg', img)


# mask_Line--二值化图（白线黑色背景）|inimg--copy后小彩图
# 返回值--是否有横线，竖线中点坐标x
# 有横线whe_row=0，只有竖线whe_row=1
def trace_oneline(mask_Line, inimg):
    h, w = mask_Line.shape[:2]
    mid_h = int(h / 2)
    cv2.line(inimg, (0, mid_h), (w, mid_h), (0, 0, 255), 2)  # #-#
    line1 = mask_Line[mid_h, :]
    position = np.where(line1 == 255)
    try:
        lp = min(position[0])
        rp = max(position[0])
        print(lp, rp)
        line_center = int((rp + lp) / 2)
        cv2.circle(inimg, (line_center, mid_h), 3, (255, 20, 0), thickness=3)  # #-#
        if (rp - lp) < w / 5:
            whe_row = 1
        else:
            whe_row = 0
    except:
        whe_row = None
        line_center = w / 2
    return whe_row, line_center


#  同上但多横线
# 返回值--是否有横线，竖线中点坐标x
# 有横线whe_row=0，只有竖线whe_row=1
def trace_multlines(mask_Line, inimg):
    h, w = mask_Line.shape[:2]
    mid_h = int(h / 2)
    cv2.line(inimg, (0, mid_h), (w, mid_h), (0, 0, 255), 2)  # #-#
    line1 = mask_Line[mid_h, :]
    position = np.where(line1 == 255)
    if position is not []:
        allLines = []
        L = []
        pp = None
        for p in position[0]:
            if pp is None or p == pp + 1:
                L.append(p)
            else:
                if len(L) >= 5:
                    allLines.append(L)
                L = [p]
            pp = p
        if len(L) >= 5:
            allLines.append(L)
        whe_row = 1
        centers = []
        for line in allLines:
            center = int((max(line) + min(line)) / 2)
            centers.append(center)
            if len(line) > w / 5:
                whe_row = 0
                cv2.circle(inimg, (center, mid_h), 3, (255, 20, 0), thickness=3)
                return whe_row, center
        for center in centers:  # #-#
            cv2.circle(inimg, (center, mid_h), 3, (255, 20, 0), thickness=3)
        if not centers:
            centers = [w / 2]
    else:
        whe_row = None
        centers = [w / 2]
    return whe_row, centers


def ser_initial(nano=None):
    if nano:
        ser_com_name = se.ser_port_open_nano()
    else:
        ser_com_name = se.ser_port_open()
    if ser_com_name is not None:
        print("baudRate: ", ser_com_name.baudrate)
        se.ser_send(ser_com_name, "-READY-")  # 已启动
    MES = MessageFormat()
    return ser_com_name, MES

def cap_initial(num=0):
    cap = cv2.VideoCapture(num)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    pro = 640/w
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h*pro)
    return cap











