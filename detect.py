# by Tiszai Istvan, GPL-3.0 license

import torch
import time
from pathlib import Path
import torch.backends.cudnn as cudnn
import cv2
from threading import (Thread, Timer)
from GUIs import *
from gc import collect
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (check_file, check_imshow, check_img_size, time_sync, non_max_suppression, xyxy2xywh, scale_coords, getRoot, model_info)
from utils.device import select_device
from models.common import DetectMultiBackend
from utils.plots import Annotator, colors

class Detect(object):
    def __init__(self, master, args):
        self.master = master
        #self.weights=getRoot(__file__)/args['weights']  # model.pt path(s)    
        self.source=getRoot(__file__)/args['source']   # file/dir/URL/glob, 0 for webcam  
        #self.data=getRoot(__file__)/args['data']   # dataset.yaml path    
        #self.imgsz=args.imgsz
        self.imgsz=(640, 640)  # inference size (height, width)
        self.conf_thres=args['conf-thres']   
        self.iou_thres=args['iou-thres']  # NMS IOU threshold    
        self.max_det=args['max-det']  # maximum detections per image   
        self.device=args['device']   # cuda device, i.e. 0 or 0,1,2,3 or cpu   
        self.view_img=args['view-img']  # show results    
        self.save_txt=args['save-txt']  # save results to *.txt    
        self.save_conf=args['save-conf'] # save confidences in --save-txt labels    
        self.save_crop=args['save-crop']  # save cropped prediction boxes    
        self.nosave=args['nosave']  # do not save images/videos    
        self.classes=args['classes']  # filter by class: --class 0, or --class 0 2 3   
        self.agnostic_nms=args['agnostic-nms']  # class-agnostic NMS    
        self.augment=args['augment']  # augmented inference    
        self.visualize=args['visualize']  # visualize features    
        #self.update=args.update  # update all models
        self.project=args['project']   # save results to project/name      
        #self.exist_ok=args.exist_ok # existing project/name ok, do not increment
        self.line_thickness=args['line-thickness'] # bounding box thickness (pixels)   
        self.hide_labels=args['hide-labels']  # hide labels   
        self.hide_conf=args['hide-conf']  # hide confidences    
        #self.half=args['half']  # use FP16 half-precision inference
        self.model=args['model']   
        self.outFlag = False
        vid_writer = None

    def resize(self, image=None, width=None, height=None):
    #dim = None
        (h, w) = image.shape[:2]
        if h<520 and w<680:
            return image
        if width is None or height is None:
            return image
        if w < h:
            r = height/float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)	

    def setStop(self,stop):
        self.outFlag = stop

    def set_modul(self, weight='weights/yolov5m6.pt', device='0', data='data/coco128.yaml', half=False):
    # Load model
        device = select_device(device)
        torch.cuda.empty_cache()
        collect()
        weight=getRoot(__file__)/weight
        data=getRoot(__file__)/data
        model = DetectMultiBackend(weight, device=device, data=data, fp16=half) 
        info = model_info(model, verbose=False, img_size=640)
        self.master.insert_textbox(message=info)
        #print (f"model name : {weight} ")
        return model

    def Start(self, modul, source):
        self.model=modul
        self.source=source
        thread = Thread(target=self.update, args=())
        thread.daemon = True  
        thread.start()

    def kill(self):       
        del self

    def write_fps(self):
        if self.dataset.mode == 'image' or self.dataset.mode == 'video' and self.model != None:
            self.master.insert_textbox(message=f'fps:{self.dt[0]:.3f},{self.dt[1]:.3f} sec\n')

    def readCam(self, cap):  
        errorCount = 0
        while cap.isOpened(): 
            if self.outFlag:
                break
            if cap.grab() == True :              
                success, im = cap.retrieve()
                if success:
                    errorCount = 0
                    self.master.image_to_frame(im)
                    #cv2.imshow("---", im)
                else:
                    print('WARNING: Video stream unresponsive, please check your camera connection.')                    
                    cap.open(0)  # re-open stream if signal was lost
                    if (errorCount > 10):
                        break;
                    errorCount+=1                        
            time.sleep(1 / 30)  # wait time
        cap.release()
        return errorCount

    def update(self):
        self.source = str(self.source)
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images       
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            self.source = check_file(self.source)  # download

     # Directories
        self.save_dir = Path(self.project)       
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(self.device)       

        if self.model != None :
            stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
            self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        else:
            stride = 32
            pt = True

        # Dataloader
        if self.webcam :
            self.view_img = check_imshow()
            if self.model == None:
                if is_url :
                    cap = cv2.VideoCapture(self.source)
                else:
                    cap = cv2.VideoCapture(0)                
                if self.readCam(cap) > 10 :
                    self.master.insert_textbox(message='Camera ERROR\n')

                return
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.master,self.source, img_size=self.imgsz, stride=stride, auto=pt)
            bs = len(self.dataset)  # batch_size
        else:
            self.dataset = LoadImages(self.master, self.source, img_size=self.imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        # Run inference
        if self.model != None :
            self.model.warmup(imgsz=(1 if pt else bs, 3, *self.imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0
 
        self.detect()   
        self.kill() #object deleting itself

    def detect(self): 
        e = 0
        for path, im, im0s, self.vid_cap, s in self.dataset:  # We can also use a for loop to iterate over our iterator class.
            cv2.waitKey(1)
            self.master.master.update_idletasks()
            #if self.dataset.mode == 'video':
            #    print(f"frame:{self.dataset.frame}")
            #else:
            #    print(f"count: {self.dataset.count}")           
            if self.outFlag == True :
                break
            self.master.clear_textbox()
            if self.model == None :      
                if self.dataset.mode == 'image' or  self.dataset.mode == 'video' :
                    im0s = self.resize(image=im0s, width=680, height=520)
                    self.master.image_to_frame(im0s)             
                continue

            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            self.visualize = self.save_dir if self.visualize else False       
            pred = self.model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            self.pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            t2 = time_sync()
            self.dt[0] = t2 - t1
           
            # Process predictions
            for i, det in enumerate(self.pred):  # per image
                t3 = time_sync()               
                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

                p = Path(p)  # to Path           
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt
                save_path = str(self.save_dir / p.name)  # im.jpg
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}\n')
                            self.master.insert_textbox(message=label)                        
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        #if self.save_crop:
                        #    save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if self.dataset.mode == 'image' or self.dataset.mode == 'video':
                    self.dt[1] = time_sync() - t3
                if self.view_img:                  
                    if self.save_img:
                        if self.dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if self.vid_path[i] != save_path:  # new video
                                self.vid_path[i] = save_path
                                if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                    self.vid_writer[i].release()  # release previous video writer
                                if self.vid_cap:  # video
                                    fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 20, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos                                
                                saveVideo(start=True, save_path=save_path, fps=fps, w=w, h=h)                                                               
                                continue;
                            cv2.waitKey(1)
                            saveVideo(image=im0)
                    im0 = self.resize(image=im0, width=680, height=520)
                    self.master.image_to_frame(im0) 

        self.write_fps()
        if self.save_img:
            if (self.dataset.mode == 'video' or  self.dataset.mode == 'stream'):           
                saveVideo(stop=True)
                self.master.start_stop(start=NORMAL, stop=NORMAL)               
        if self.view_img:      
            if self.dataset.mode == 'stream':
                if self.vid_cap != None:
                    self.vid_cap.release()

vid_writer=None
vs_timer=None
vs_start=False
vs_stop=False
vs_save_path=''
vs_fps=0
vs_w=0 
vs_h=0
vs_image=None

def saveVideo(start=False, stop=False, image=None, save_path='', fps=0, w=0, h=0): 
    global vs_timer,vs_start,vs_stop,vs_save_path,vs_fps,vs_w,vs_h,vs_image
    vs_start=start
    vs_stop=stop
    if vs_start :
        vs_save_path=save_path
        vs_fps=fps
        vs_w=w
        vs_h=h
    elif vs_stop == False :
        vs_image=image
    vs_timer = Timer(interval=0.001, function=timer_handler)
    vs_timer.start()

def timer_handler():
    global vs_start,vs_stop,vs_save_path,vs_fps,vs_w,vs_h,vs_image
    global vid_writer
    if vs_start :
        vid_writer = cv2.VideoWriter(vs_save_path, cv2.VideoWriter_fourcc(*'mp4v'), vs_fps, (vs_w,vs_h)) 
    elif vs_stop:
        if vid_writer != None:
            vid_writer.release() 
            vid_writer = None
    else:
        if vid_writer != None:
            if vid_writer.isOpened() == True:
                vid_writer.write(vs_image)
    vs_timer.cancel()