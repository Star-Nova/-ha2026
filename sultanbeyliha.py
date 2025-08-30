#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
İHA ALL-IN-ONE v3
Güncellenmiş, sağlam ve yarışma-dostu nesne tespit sistemi.

Öne Çıkanlar (v3):
- Roboflow indirme/kurulum iş akışı yeniden yazıldı (URL/ID fark etmez, hatalara dayanıklı).
- Kamera kaynak seçimi kullanıcı dostu: 0 (dahili), 1 (harici), ya da RTSP/HTTP IP akışı.
- Performans profilleri (speed/balanced/accuracy) + akıllı çerçeve atlama.
- Adaptif renk tespiti (aydınlık/kontrast normalize), gelişmiş şekil tespiti korumaları.
- Dummy (simülasyon) GPS/MAVLink modları: Donanım olmadan test edilebilir.
- Kayıt yönetimi: Dosya döndürme (rotation) ve üst sınırlar.
- GUI var/yok otomatik algı: headless ortamda da problemsiz.
- Kod modüler, tek dosyada ama net bölümlenmiş.

Notlar:
- YOLOv8 için `ultralytics` gerekir: pip install ultralytics
- OCR (opsiyonel) için: pip install easyocr
- Roboflow (opsiyonel) için: pip install roboflow
- GPS için: pip install pyserial pynmea2
- MAVLink için: pip install pymavlink

"""

import os
import sys
import time
import json
import socket
import requests
import shutil
import glob
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import cv2

# ============== Opsiyonel Kütüphaneler ==============
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics (YOLOv8) yüklü olmalı. pip install ultralytics") from e

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except Exception:
    ROBOFLOW_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

try:
    import serial, pynmea2
    GPS_LIBS_AVAILABLE = True
except Exception:
    GPS_LIBS_AVAILABLE = False

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except Exception:
    MAVLINK_AVAILABLE = False

# ============== GUI Kontrol ==============

def gui_available() -> bool:
    try:
        name = "__cv_test_win__"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow(name)
        return True
    except Exception:
        return False

GUI = gui_available()


def safe_imshow(name, img):
    if GUI:
        try:
            cv2.imshow(name, img)
        except Exception:
            pass


def safe_waitkey(delay=1):
    if GUI:
        try:
            return cv2.waitKey(delay) & 0xFF
        except Exception:
            return -1
    return -1


def safe_destroy_all():
    if GUI:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

# ============== Genel Yardımcılar ==============

def get_local_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "unknown_local"


def get_public_ip(timeout=2):
    try:
        return requests.get("https://api.ipify.org", timeout=timeout).text
    except Exception:
        return "unknown_public"


# ============== Roboflow Yardımcı (Daha Dayanıklı) ==============

def _parse_rf_url(workspace_or_url: str, project: str | None) -> tuple[str, str]:
    """workspace ve project'i güvenle çöz.
    - workspace_or_url bir URL ise (örn: https://universe.roboflow.com/ws/proj),
      otomatik ayrıştırılır.
    - project None ise URL'den alınır.
    """
    ws = workspace_or_url
    pj = project
    if workspace_or_url.startswith("http"):
        # URL beklenen formatlar: universe.roboflow.com/<ws>/<pj>
        parts = [p for p in workspace_or_url.split("/") if p]
        # en sonda proje adı, ondan önce workspace varmış gibi varsayalım
        if len(parts) >= 2:
            pj = parts[-1]
            ws = parts[-2]
    if not ws or not pj:
        raise ValueError("Roboflow workspace/project bilgisi eksik veya çözümlenemedi.")
    return ws, pj


def roboflow_download(api_key: str, workspace_or_url: str, project: str | None = None,
                      version: int = 1, fmt: str = "yolov8", out_dir: str = "roboflow_download"):
    """Roboflow veri setini indir. Çökmeden, anlaşılır hata mesajlarıyla.
    - workspace_or_url: "weagle" veya tam URL.
    - project: "weagle-iha-nesne-tespiti" ya da URL verildiyse None bırak.
    - out_dir: her zaman yerel klasör olmalıdır (IP/URL verilmeyecek!).
    """
    if not ROBOFLOW_AVAILABLE:
        print("[RF] Roboflow SDK yüklü değil. (pip install roboflow)")
        return None
    try:
        if os.path.sep not in out_dir and (out_dir.startswith("http") or out_dir.replace(".", "").isdigit()):
            # Kullanıcı yanlışlıkla IP/URL verdiyse engelle
            raise ValueError("Çıkış konumu yerel klasör olmalı (ör: roboflow_download). IP/URL verilemez.")
        os.makedirs(out_dir, exist_ok=True)
        ws, pj = _parse_rf_url(workspace_or_url, project)
        print(f"[RF] Workspace: {ws} | Project: {pj} | Version: {version} | Format: {fmt}")
        rf = Roboflow(api_key=api_key)
        prj = rf.workspace(ws).project(pj)
        ver = prj.version(version)
        print("[RF] İndiriliyor... (bağlantı hızına göre sürebilir)")
        ds = ver.download(fmt, location=out_dir)
        print("[RF] İndirildi →", ds)
        return ds
    except Exception as e:
        print("[RF] Hata:", e)
        print("[RF] Kontrol listesi: API key doğru mu? Workspace/Project adı doğru mu? İnternet erişimi var mı?")
        return None


# ============== GPS / MAVLink ==============

class GPSReader:
    def __init__(self, port: str | None, baud: int = 9600, timeout: int = 1, simulate: bool = False):
        self.last = None
        self.serial = None
        self.sim = simulate or (not GPS_LIBS_AVAILABLE) or (not port)
        if self.sim:
            print("[GPS] Simülasyon modu açık.")
        else:
            try:
                self.serial = serial.Serial(port, baud, timeout=timeout)
                print(f"[GPS] Bağlı: {port}")
            except Exception as e:
                print("[GPS] Açılamadı, simülasyona geçiliyor:", e)
                self.sim = True

    def read(self):
        if self.sim:
            # Basit bir sinüs/örnek değer üretimi
            t = time.time()
            lat = 40.99 + 0.0001 * np.sin(t / 10.0)
            lon = 29.12 + 0.0001 * np.cos(t / 10.0)
            alt = 35.0
            self.last = (lat, lon, alt)
            return self.last
        if not self.serial:
            return None
        try:
            line = self.serial.readline().decode('ascii', errors='ignore')
            if line.startswith(("$GPGGA", "$GPRMC")):
                msg = pynmea2.parse(line)
                lat = getattr(msg, 'latitude', None)
                lon = getattr(msg, 'longitude', None)
                alt = getattr(msg, 'altitude', None)
                self.last = (lat, lon, alt)
                return self.last
        except Exception:
            return self.last
        return self.last

    def close(self):
        if self.serial:
            try:
                self.serial.close()
            except Exception:
                pass


class MAVLinkSender:
    def __init__(self, conn_str: str | None, simulate: bool = False):
        self.conn = None
        self.sim = simulate or (not MAVLINK_AVAILABLE) or (not conn_str)
        if self.sim:
            print("[MAV] Simülasyon modu açık.")
            return
        try:
            self.conn = mavutil.mavlink_connection(conn_str)
            self.conn.wait_heartbeat(timeout=5)
            print("[MAV] Bağlandı:", conn_str)
        except Exception as e:
            print("[MAV] Hata, simülasyona geçiliyor:", e)
            self.sim = True

    def send_text(self, txt: str) -> bool:
        if self.sim:
            # Simde sadece konsola yaz
            print("[MAV] (SIM) →", txt)
            return True
        if not self.conn:
            return False
        try:
            self.conn.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_NOTICE, txt.encode())
            return True
        except Exception as e:
            print("[MAV] Gönderim hatası:", e)
            return False

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass


# ============== Görsel Yardımcı: Etiket ==============

def draw_label_with_background(img, text, topleft, font=cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale=0.6, thickness=1, padding=6,
                               bg_color=(20, 20, 20), bg_alpha=0.6, text_color=(255, 255, 255)):
    x, y = topleft
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    rect_w = w + 2 * padding
    rect_h = h + 2 * padding
    rect_x1 = x
    rect_y1 = y - rect_h
    rect_x2 = x + rect_w
    rect_y2 = y

    H, W = img.shape[:2]
    if rect_y1 < 0:
        rect_y1 = y
        rect_y2 = y + rect_h

    overlay = img.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), 1)

    text_x = rect_x1 + padding
    text_y = rect_y2 - padding - baseline
    cv2.putText(img, text, (text_x + 1, text_y + 1), font, font_scale, (10, 10, 10), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# ============== Ana Detector Sınıfı ==============

class IhaDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, perf_mode: str = "balanced",
                 enable_ocr: bool = False, enable_gps: bool = False, gps_port: str | None = None,
                 enable_mav: bool = False, mav_conn: str | None = None, simulate_gps: bool = False,
                 simulate_mav: bool = False, output_root: str | None = None, device: str | None = None,
                 max_json: int = 200, max_images: int = 200, frame_skip: int | None = None):

        presets = {
            "speed": {"imgsz": 416, "save_every": 20, "ocr_every": 8, "default_skip": 1},
            "balanced": {"imgsz": 640, "save_every": 6, "ocr_every": 4, "default_skip": 0},
            "accuracy": {"imgsz": 1024, "save_every": 1, "ocr_every": 1, "default_skip": 0},
        }
        p = presets.get(perf_mode, presets["balanced"])
        self.imgsz = p["imgsz"]
        self.save_every = p["save_every"]
        self.ocr_every = p["ocr_every"]
        self.frame_skip = p["default_skip"] if frame_skip is None else max(0, int(frame_skip))

        print("[YOLO] Model yükleniyor:", model_path)
        self.model = YOLO(model_path)
        self.conf_thres = conf
        self.names = self.model.names if hasattr(self.model, "names") else {0: "object"}
        self.device = device  # None = otomatik

        self.frame_id = 0
        self.start_time = time.time()
        self.counts = defaultdict(int)

        # OCR
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr = None
        if self.enable_ocr:
            try:
                self.ocr = easyocr.Reader(['tr', 'en'])
                print("[OCR] Etkin.")
            except Exception as e:
                print("[OCR] Başlatılamadı:", e)
                self.enable_ocr = False

        # GPS
        self.gps = GPSReader(gps_port, simulate=simulate_gps) if enable_gps else None

        # MAVLink
        self.mav = MAVLinkSender(mav_conn, simulate=simulate_mav) if enable_mav else None

        # kayıt
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = output_root or f"iha_session_{ts}"
        os.makedirs(self.root, exist_ok=True)
        self.img_dir = os.path.join(self.root, "annotated_images"); os.makedirs(self.img_dir, exist_ok=True)
        self.json_dir = os.path.join(self.root, "json"); os.makedirs(self.json_dir, exist_ok=True)
        self.csv_path = os.path.join(self.root, "detections.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                f.write("timestamp,frame,class,conf,x1,y1,x2,y2,cx,cy,area,color,shape,text,gps_lat,gps_lon,gps_alt\n")

        # log sınırlamaları
        self.max_json = max_json
        self.max_images = max_images
        self._recent_json = deque(sorted(glob.glob(os.path.join(self.json_dir, "*.json"))))
        self._recent_imgs = deque(sorted(glob.glob(os.path.join(self.img_dir, "*.jpg"))))

        # renk aralıkları (HSV)
        self.base_color_ranges = {
            "kirmizi": [(0, 70, 50), (10, 255, 255), (170, 70, 50), (180, 255, 255)],
            "mavi": [(100, 120, 50), (130, 255, 255)],
            "yesil": [(40, 40, 40), (80, 255, 255)],
            "sari": [(15, 100, 100), (35, 255, 255)],
        }

        print("[SYS] Başlatıldı. Local IP:", get_local_ip(), "Public IP:", get_public_ip())

    # ---------- Yardımcılar ----------
    def _rotate_files(self, path_list: deque, limit: int):
        while len(path_list) > limit:
            p = path_list.popleft()
            try:
                os.remove(p)
            except Exception:
                pass

    # Adaptif renk için V kanalına göre eşik genişletme
    def _adapt_ranges(self, hsv_roi):
        v_mean = hsv_roi[..., 2].mean() if hsv_roi.size else 128.0
        scale = 1.0
        if v_mean < 80:
            scale = 0.8  # karanlıkta aralıkları biraz genişlet
        elif v_mean > 180:
            scale = 1.1  # çok aydınlıkta hafif genişlet
        ranges = {}
        for name, r in self.base_color_ranges.items():
            if name == "kirmizi" and len(r) == 4:
                r0 = (r[0][0], int(r[0][1] * scale), int(r[0][2] * scale))
                r1 = (r[1][0], int(r[1][1] * scale), int(r[1][2] * scale))
                r2 = (r[2][0], int(r[2][1] * scale), int(r[2][2] * scale))
                r3 = (r[3][0], int(r[3][1] * scale), int(r[3][2] * scale))
                ranges[name] = [r0, r1, r2, r3]
            else:
                r0 = (r[0][0], int(r[0][1] * scale), int(r[0][2] * scale))
                r1 = (r[1][0], int(r[1][1] * scale), int(r[1][2] * scale))
                ranges[name] = [r0, r1]
        return ranges

    # ---------- Renk / Şekil / OCR ----------
    def detect_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        H, W = frame.shape[:2]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "bilinmeyen", 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ranges = self._adapt_ranges(hsv)
        total = hsv.shape[0] * hsv.shape[1]
        best = ("belirsiz", 0.0)
        for name, r in ranges.items():
            if name == "kirmizi" and len(r) == 4:
                mask1 = cv2.inRange(hsv, np.array(r[0]), np.array(r[1]))
                mask2 = cv2.inRange(hsv, np.array(r[2]), np.array(r[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, np.array(r[0]), np.array(r[1]))
            cnt = int(np.count_nonzero(mask))
            score = cnt / float(total) if total > 0 else 0.0
            if score > best[1]:
                best = (name, score)
        return best if best[1] > 0.07 else ("belirsiz", best[1])

    def detect_shape(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        H, W = frame.shape[:2]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "belirsiz"
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # ışık değişimlerine karşı
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return "belirsiz"
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 50:
            return "kucuk"
        eps = 0.04 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        v = len(approx)
        if v == 3:
            return "ucgen"
        if v == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = float(w) / h if h > 0 else 0
            return "kare" if 0.9 <= ar <= 1.1 else "dikdortgen"
        if v > 8:
            return "daire"
        return f"{v}gen"

    def read_text(self, frame, bbox):
        if not self.enable_ocr or self.ocr is None:
            return ""
        if (self.frame_id % self.ocr_every) != 0:
            return ""
        x1, y1, x2, y2 = map(int, bbox)
        H, W = frame.shape[:2]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return ""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 7, 50, 50)
            res = self.ocr.readtext(gray, detail=0, paragraph=True)
            if res:
                txt = max(res, key=len).strip()
                txt = ''.join(c for c in txt if c.isalnum() or c.isspace())
                return txt
        except Exception:
            return ""
        return ""

    # ---------- Ana İşleme ----------
    def process_frame(self, frame):
        self.frame_id += 1
        if self.frame_skip and (self.frame_id % (self.frame_skip + 1) != 1):
            # Basit çerçeve atlama
            annotated = frame.copy()
            self.draw_overlay(annotated, fps=0.0, det_count=0)
            return annotated, []

        t0 = time.time()
        h, w = frame.shape[:2]
        maxd = max(h, w)
        if maxd > self.imgsz:
            scale = self.imgsz / float(maxd)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # YOLO inference
        results = self.model(frame, conf=self.conf_thres, verbose=False, device=self.device)
        detections = []
        gps = self.gps.read() if self.gps else None

        for r in results:
            if getattr(r, 'boxes', None) is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                cls_name = self.names.get(cls, f"class_{cls}")
                x1, y1, x2, y2 = map(int, xyxy)
                area = (x2 - x1) * (y2 - y1)
                if area < 80:
                    continue

                color, color_conf = self.detect_color(frame, (x1, y1, x2, y2))
                shape = self.detect_shape(frame, (x1, y1, x2, y2))
                text = self.read_text(frame, (x1, y1, x2, y2))

                det = {
                    "timestamp": datetime.now().isoformat(),
                    "frame": self.frame_id,
                    "class": cls_name,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    "area": area,
                    "color": color,
                    "color_conf": float(color_conf),
                    "shape": shape,
                    "text": text,
                    "gps": gps,
                }
                detections.append(det)
                self.draw_detection(frame, det)

                key = f"{cls_name}|{color}|{shape}"
                self.counts[key] += 1

                # MAVLink kısa bildirim (opsiyonel)
                if self.mav and gps and gps[0] and gps[1]:
                    msg = f"DETECT:{cls_name} LAT:{gps[0]:.6f} LON:{gps[1]:.6f}"
                    self.mav.send_text(msg)

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        self.draw_overlay(frame, fps, len(detections))
        if detections and (self.frame_id % self.save_every == 0):
            self.save_detections(detections, frame)
        return frame, detections

    # ---------- Çizimler ----------
    def draw_detection(self, frame, det):
        x1, y1, x2, y2 = det["bbox"]
        col_map = {
            "kirmizi": (0, 0, 255),
            "mavi": (255, 0, 0),
            "yesil": (0, 255, 0),
            "sari": (0, 255, 255),
            "belirsiz": (180, 180, 180),
        }
        base_color = col_map.get(det["color"], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, 2)
        label = f"{det['class']} {det['conf']:.2f} | {det['color']} {det['shape']}"
        if det["text"]:
            label += f" '{det['text'][:12]}'"
        draw_label_with_background(frame, label, (x1, y1), font_scale=0.55, thickness=1,
                                   bg_color=(30, 30, 30), bg_alpha=0.65, text_color=(255, 255, 255))

    def draw_overlay(self, frame, fps, det_count):
        H, W = frame.shape[:2]
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_id}",
            f"Det: {det_count}",
        ]
        panel_w = 220
        panel_h = 20 * len(info_lines) + 16
        panel_x1 = 8
        panel_y1 = 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x1 + panel_w, panel_y1 + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for i, l in enumerate(info_lines):
            cv2.putText(frame, l, (panel_x1 + 8, panel_y1 + 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 255, 200), 1, cv2.LINE_AA)

    # ---------- Kayıt ----------
    def save_detections(self, detections, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jpath = os.path.join(self.json_dir, f"detections_{ts}.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(detections, f, ensure_ascii=False, indent=2)
        imgpath = os.path.join(self.img_dir, f"annot_{ts}.jpg")
        cv2.imwrite(imgpath, frame)
        # CSV append
        with open(self.csv_path, "a", encoding="utf-8") as f:
            for d in detections:
                gps = d.get("gps") or (None, None, None)
                safe_text = (d.get("text") or "").replace('"', '')
                line = ",".join([
                    d["timestamp"], str(d["frame"]), d["class"], f"{d['conf']:.3f}",
                    str(d["bbox"][0]), str(d["bbox"][1]), str(d["bbox"][2]), str(d["bbox"][3]),
                    str(d["center"][0]), str(d["center"][1]), str(d["area"]),
                    d["color"], d["shape"], f'"{safe_text}"',
                    str(gps[0] if len(gps) > 0 else ""), str(gps[1] if len(gps) > 1 else ""),
                    str(gps[2] if len(gps) > 2 else ""),
                ])
                f.write(line + "\n")
        # Dosya döndürme
        self._recent_json.append(jpath)
        self._recent_imgs.append(imgpath)
        self._rotate_files(self._recent_json, self.max_json)
        self._rotate_files(self._recent_imgs, self.max_images)
        print(f"[LOG] Kayıt: JSON:{jpath} IMG:{imgpath}")

    def final_report(self):
        elapsed = time.time() - self.start_time
        avg = self.frame_id / elapsed if elapsed > 0 else 0
        print("\n--- OTURUM ÖZETİ ---")
        print(f"Süre: {elapsed:.1f}s, Frame: {self.frame_id}, Ortalama FPS: {avg:.2f}")
        print("En çok tespitler (ilk 20):")
        for k, v in sorted(self.counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {k}: {v}")
        print("Kayıt dizin:", self.root)
        print("--------------------")

    # ---------- Çalışma Döngüleri ----------
    def run_camera(self, source=0, show_original=True, window_name="IHA Detector"):
        print("[CAM] Başlatılıyor:", source)
        cap = cv2.VideoCapture(source)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            print("[CAM] Açılamadı.")
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[CAM] Frame okunamadı.")
                    break
                annotated, dets = self.process_frame(frame.copy())
                if show_original:
                    safe_imshow("Original", frame)
                safe_imshow(window_name, annotated)
                key = safe_waitkey(1)
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    p = os.path.join(self.img_dir, f"manual_{ts}.jpg")
                    cv2.imwrite(p, annotated); print("[CAM] Kaydedildi:", p)
                elif key == ord('r'):
                    print("[CAM] Sayaçlar:", dict(self.counts))
        except KeyboardInterrupt:
            print("[CAM] İptal edildi.")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if self.mav:
                self.mav.close()
            if self.gps:
                self.gps.close()
            safe_destroy_all()
            self.final_report()

    def run_video(self, path, output=None, show_original=False):
        print("[VID] İşleniyor:", path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("[VID] Açılamadı.")
            return
        out = None
        if output:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output, fourcc, fps, (w, h))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, dets = self.process_frame(frame.copy())
                if out:
                    out.write(annotated)
                if show_original:
                    safe_imshow("Original", frame)
                safe_imshow("Annotated", annotated)
                if self.frame_id % 30 == 0:
                    print("[VID] Frames:", self.frame_id)
                if GUI and safe_waitkey(1) == ord('q'):
                    break
        finally:
            cap.release()
            if out:
                out.release()
            safe_destroy_all()
            self.final_report()

    def run_demo_images(self, folder, delay=1000):
        imgs = sorted([p for p in glob.glob(os.path.join(folder, "**/*.*"), recursive=True)
                       if p.lower().endswith((".jpg", ".jpeg", ".png"))])
        if not imgs:
            print("[DEMO] Klasörde görüntü yok.")
            return
        print(f"[DEMO] {len(imgs)} görüntü, her biri {delay}ms")
        for pth in imgs:
            img = cv2.imread(pth)
            if img is None:
                continue
            annotated, dets = self.process_frame(img.copy())
            safe_imshow("Demo Original", img)
            safe_imshow("Demo Annotated", annotated)
            print("[DEMO] Dosya:", os.path.basename(pth), "Tespit:", [d['class'] for d in dets])
            k = safe_waitkey(delay)
            if k == ord('q'):
                break
        safe_destroy_all()
        self.final_report()


# ============== CLI ==============

def main():
    print("=== İHA ALL-IN-ONE v3 ===")
    print("Local IP:", get_local_ip(), "Public IP:", get_public_ip())
    while True:
        print("\nSeçenekler:")
        print("1) Kamera (webcam / USB / IP)")
        print("2) Video dosyası")
        print("3) Demo (klasörden resim test)")
        print("4) Roboflow dataset indir")
        print("0) Çıkış")
        c = input("Seçiminiz: ").strip()
        if c == "0":
            break
        if c == "1":
            print("\nKamera portunu seçiniz:")
            print("0 = Dahili kamera (laptop)")
            print("1 = Harici USB kamera")
            src_in = input("Seçiminiz (0/1 veya rtsp://...): ").strip() or "0"
            try:
                src_eval = int(src_in)
            except Exception:
                src_eval = src_in  # rtsp/http olabilir

            model = input("Model yolu (Enter=yolov8n.pt): ").strip() or "yolov8n.pt"
            perf = input("Performans (speed/balanced/accuracy) [balanced]: ").strip() or "balanced"
            ocr_flag = input("OCR? (y/n) [n]: ").strip().lower() == 'y'
            gps_flag = input("GPS? (y/n) [n]: ").strip().lower() == 'y'
            simulate_gps = False
            gps_port = None
            if gps_flag:
                gps_port = input("GPS seri port (ör: COM3 veya /dev/ttyUSB0) [SIM için boş bırak]: ").strip() or None
                simulate_gps = gps_port is None
            mav_flag = input("MAVLink? (y/n) [n]: ").strip().lower() == 'y'
            simulate_mav = False
            mav_conn = None
            if mav_flag:
                mav_conn = input("MAVLink conn (ör udp:127.0.0.1:14550) [SIM için boş bırak]: ").strip() or None
                simulate_mav = mav_conn is None

            det = IhaDetector(model_path=model, conf=0.35, perf_mode=perf,
                              enable_ocr=ocr_flag, enable_gps=gps_flag, gps_port=gps_port,
                              enable_mav=mav_flag, mav_conn=mav_conn,
                              simulate_gps=simulate_gps, simulate_mav=simulate_mav)
            det.run_camera(source=src_eval, show_original=True)

        elif c == "2":
            vp = input("Video yolu: ").strip()
            if not os.path.exists(vp):
                print("[VID] Dosya bulunamadı.")
                continue
            model = input("Model (Enter=yolov8n.pt): ").strip() or "yolov8n.pt"
            det = IhaDetector(model_path=model, perf_mode="balanced", enable_ocr=False)
            outp = input("Çıkış video dosyası (opsiyonel): ").strip() or None
            det.run_video(vp, output=outp, show_original=False)

        elif c == "3":
            folder = input("Demo klasör yolu (görüntüler): ").strip() or "."
            model = input("Model (Enter=yolov8n.pt): ").strip() or "yolov8n.pt"
            det = IhaDetector(model_path=model, perf_mode="balanced", enable_ocr=False)
            delay = int(input("Görüntü başına ms [1200]: ").strip() or "1200")
            det.run_demo_images(folder, delay=delay)

        elif c == "4":
            if not ROBOFLOW_AVAILABLE:
                print("[RF] Roboflow SDK yüklü değil (pip install roboflow)")
                continue
            api = input("Roboflow API key: ").strip()
            ws_or_url = input("Workspace veya URL (ör: weagle veya https://universe.roboflow.com/ws/proj): ").strip()
            pj = input("Project (URL verdiyseniz boş bırakın): ").strip() or None
            ver = int(input("Version [1]: ").strip() or "1")
            out = input("Çıkış klasörü [roboflow_download]: ").strip() or "roboflow_download"
            fmt = input("Format [yolov8]: ").strip() or "yolov8"
            roboflow_download(api, ws_or_url, pj, ver, fmt, out)
        else:
            print("Geçersiz seçim.")
    print("Çıkış yapılıyor.")


if __name__ == "__main__":
    main()
