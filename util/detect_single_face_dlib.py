import dlib

#https://hackernoon.com/gender-and-race-change-on-your-selfie-with-neural-nets-9a9a1c9c5c16
def detect_single_face_dlib(img_rgb, rescale=(1.1, 1.5, 1.1, 1.3)):
    fd_front_dlib = dlib.get_frontal_face_detector()
    face = fd_front_dlib(img_rgb, 1)
    if len(face) > 0:
        face = sorted([(t.width() * t.height(), (t.left(), t.top(), t.width(), t.height()))
                       for t in face],
                      key=lambda t: t[0], reverse=True)[0][1]
    else:
        return None

    if rescale is not None and face is not None:
        if type(rescale) != tuple:
            rescale = (rescale, rescale, rescale, rescale)
        (x, y, w, h) = face

        w = min(img_rgb.shape[1] - x, int(w / 2 + rescale[2] * w / 2))
        h = min(img_rgb.shape[0] - y, int(h / 2 + rescale[3] * h / 2))

        fx = max(0, int(x + w / 2 * (1 - rescale[0])))
        fy = max(0, int(y + h / 2 * (1 - rescale[1])))
        fw = min(img_rgb.shape[1] - fx, int(w - w / 2 * (1 - rescale[0])))
        fh = min(img_rgb.shape[0] - fy, int(h - h / 2 * (1 - rescale[1])))

        face = (fx, fy, fw, fh)
    return face
